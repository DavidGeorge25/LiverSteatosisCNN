"""Provenance stamping.

Any number in a paper has to be reproducible from the artifact that produced
it. This records the code version, the environment, the config, and the seeds
alongside every output directory, so a result can be traced back to the exact
state that generated it -- including whether the working tree was dirty at the
time, which is the failure mode that silently invalidates "reproducible" runs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PACKAGES = [
    "numpy", "pandas", "opencv-python-headless", "scikit-image",
    "openslide-python", "scipy", "pyyaml", "matplotlib",
]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def git_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the commit hash does not describe the code that
        # ran. Recorded explicitly rather than silently ignored.
        "dirty": bool(status) if status is not None else None,
        "dirty_files": status.splitlines() if status else [],
    }


def package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    try:
        import openslide

        out["openslide-c-library"] = openslide.__library_version__
    except Exception:
        pass
    return out


def file_digest(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed -- these are 250 MB slides."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def config_digest(cfg: Any) -> str:
    """Stable hash of a resolved config, so runs can be grouped by parameters."""
    payload = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def stamp(
    out_dir: str | Path,
    cfg: Any = None,
    extra: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    hash_inputs: bool = False,
) -> Path:
    """Write `provenance.json` into `out_dir`.

    `hash_inputs` is off by default: digesting a 9 GB cohort takes minutes and
    is only worth it for a final archival run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "command": " ".join(sys.argv),
    }
    if cfg is not None:
        record["config_digest"] = config_digest(cfg)
        record["config"] = cfg.to_dict()
    if inputs:
        record["inputs"] = [
            {"path": p, "sha256": file_digest(p) if hash_inputs else None}
            for p in inputs
        ]
    if extra:
        record.update(extra)

    path = out_dir / "provenance.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path
