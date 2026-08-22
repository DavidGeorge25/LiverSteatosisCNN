"""Package a labelling set as a folder a pathologist can run unaided.

The review app is stdlib-only by design, but living inside `mashpath` it is
reachable only through package `__init__`s that import the candidate-review
flow, and that pulls NumPy, OpenCV and OpenSlide in behind it. Asking a
pathologist to install OpenSlide before she can look at pictures is how a
labelling round quietly does not happen.

So a bundle is the four modules the app actually needs, a minimal package shim
in place of the real `__init__`, the set itself, her brief, and a launcher per
platform. Nothing is rewritten -- the modules are copied verbatim, so the
bundle cannot drift from what is tested here.

    python -m mashpath.review.bundle review_sets/ballooning_tiles_pilot \
        --out ~/Desktop/ballooning_labelling
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .names import MANIFEST_NAME

# Only these. Every one is stdlib-only; `bundle` itself is not included
# because a bundle does not build bundles.
MODULES = ("names.py", "verdicts.py", "tiles_app.py")

SHIM = '''"""Minimal package shim for the standalone labelling bundle.

Deliberately empty. The real mashpath.review.__init__ imports the candidate
flow, which needs NumPy/OpenCV/OpenSlide; the labelling app needs none of that
and must import without them.
"""
'''

MAC = """#!/bin/bash
# Double-click this file. It opens the labelling app in your browser.
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "Python 3 was not found. Install it from python.org, then try again."
  read -r -p "Press return to close."
  exit 1
fi
"$PY" -m review.tiles_app "{set_name}" --port {port} &
sleep 2
open "http://localhost:{port}/"
echo
echo "The app is running. Leave this window OPEN while you work."
echo "Close it when you are finished -- your answers are already saved."
wait
"""

WIN = """@echo off
cd /d "%~dp0"
where python >nul 2>nul || (
  echo Python 3 was not found. Install it from python.org, then try again.
  pause
  exit /b 1
)
start "" http://localhost:{port}/
python -m review.tiles_app "{set_name}" --port {port}
pause
"""

READ_ME = """# Ballooning labelling — start here

Everything you need is in this folder. Nothing to install.

## To start

- **macOS**: double-click `label_mac.command`
- **Windows**: double-click `label_windows.bat`

Your browser opens on the first field. If it does not, open
<http://localhost:{port}/> yourself.

Leave the black terminal window open while you work — that is the program.
Closing it stops the app; **it does not lose anything**, because every answer
is written to disk the moment you press a key.

## To stop, and to resume

Close the window whenever you like. To carry on, start it the same way and
type the same initials — it resumes at the first field you have not answered.

## Your answers

They accumulate in `{set_name}/verdicts.csv`. When you are done, send that one
file back. It is small, and it is the only file that matters — everything else
in this folder can be regenerated.

## What to do

`BRIEF.md` in this folder is the one-page description of the task. Please read
it before you start; it takes two minutes.
"""


def build(package: Path | str, out: Path | str, port: int = 8000,
          brief: Path | str | None = None, verbose: bool = True) -> Path:
    """Copy `package` and the app into a self-contained folder at `out`."""
    package, out = Path(package), Path(out)
    if not (package / MANIFEST_NAME).exists():
        raise FileNotFoundError(f"{package} has no {MANIFEST_NAME}")
    out.mkdir(parents=True, exist_ok=True)

    src = Path(__file__).parent
    app = out / "review"
    app.mkdir(exist_ok=True)
    (app / "__init__.py").write_text(SHIM)
    for m in MODULES:
        shutil.copy2(src / m, app / m)

    dest = out / package.name
    if dest.exists():
        shutil.rmtree(dest)
    # The frame carries score/band/split. It is the analysis record and has no
    # business in the reviewer's copy -- a spreadsheet left open at the wrong
    # column is exactly the anchoring the app is built to avoid.
    shutil.copytree(package, dest,
                    ignore=shutil.ignore_patterns("sampling_frame.csv",
                                                  "sampling_report.md",
                                                  "verdicts.csv"))

    fmt = {"set_name": package.name, "port": port}
    (out / "label_mac.command").write_text(MAC.format(**fmt))
    (out / "label_mac.command").chmod(0o755)
    (out / "label_windows.bat").write_text(WIN.format(**fmt))
    (out / "README.md").write_text(READ_ME.format(**fmt))
    if brief and Path(brief).exists():
        shutil.copy2(brief, out / "BRIEF.md")

    if verbose:
        n = sum(1 for _ in (dest / "images").glob("*")) if (dest / "images").is_dir() else 0
        mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
        print(f"bundle -> {out}")
        print(f"  {package.name}: {n} fields")
        print(f"  app: {', '.join(MODULES)} (stdlib only)")
        print(f"  size: {mb:.0f} MB")
        print(f"  excluded: sampling_frame.csv, sampling_report.md "
              f"(score/band/split -- analysis only)")
    return out


def _main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m mashpath.review.bundle",
        description="Package a labelling set for a reviewer's own machine.")
    p.add_argument("package", help="a built review_sets/<name> directory")
    p.add_argument("--out", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--brief", default="docs/BALLOONING_TILE_BRIEF.md")
    a = p.parse_args(argv)
    build(a.package, a.out, a.port, a.brief)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
