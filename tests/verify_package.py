"""Pre-ship checks on a built annotation package. Run it before sending anything.

    python tests/verify_package.py ~/Desktop/Ballooning_Study review_sets/ballooning_full
    python tests/verify_package.py ~/Desktop/Ballooning_Warmup review_sets/ballooning_warmup

Checks the ARTEFACT, not the source that produced it. Every failure this is
written against has happened at least once in a project like this one: a page
built from a stale template that claimed the wrong number of sections, an image
folder carrying renders from a previous draw, a launcher that worked in the
directory it was written in, a font that only loaded on a machine with internet.
None of those are visible in a diff.

The browser flow is driven through the SHIPPED script, extracted back out of
index.html, so what is tested is what she will run.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((name, bool(ok), detail))
    return bool(ok)


def extract_script(index_html: str) -> str:
    """The page's own JS, back out of the built page."""
    blocks = re.findall(r"<script>\n(.*?)</script>", index_html, re.S)
    if not blocks:
        raise ValueError("no plain <script> block in index.html")
    return blocks[-1]


def verify(pkg: Path, work: Path) -> None:
    pkg, work = Path(pkg).expanduser(), Path(work).expanduser()

    # ---- 1. the files she needs ------------------------------------------
    web = (pkg / "robots.txt").exists()
    needed = ["index.html", "README.txt"]
    needed += ["robots.txt"] if web else ["START_MAC.command", "START_WINDOWS.bat"]
    for name in needed:
        check(f"present: {name}", (pkg / name).exists())
    if web:
        check("hosted build carries no launcher to double-click",
              not (pkg / "START_MAC.command").exists())
        check("hosted build asks not to be indexed",
              'content="noindex' in (pkg / "index.html").read_text())
        check("README does not tell her to double-click anything",
              "double-click" not in (pkg / "README.txt").read_text().lower())
    else:
        launcher = pkg / "START_MAC.command"
        check("START_MAC.command is executable",
              launcher.exists() and os.access(launcher, os.X_OK))

    html = (pkg / "index.html").read_text()
    key = pd.read_csv(work / "key.csv")

    # ---- 2. the draw -----------------------------------------------------
    strata = set(key["sample_type"])
    check("no `enriched` rows in key.csv", "enriched" not in strata,
          f"strata present: {sorted(strata)}")
    check("every field is uniform or a repeat", strata <= {"uniform", "repeat"},
          f"strata present: {sorted(strata)}")
    rep = (key["sample_type"] == "repeat").mean()
    check("repeat share is 8-13%", 0.08 <= rep <= 0.13, f"{100 * rep:.1f}%")
    per_section = key.groupby("blind_id").size()
    check("every section has the same number of fields",
          per_section.nunique() == 1, f"sizes: {sorted(set(per_section))}")
    shares = key.groupby("batch").size() / len(key)
    # The cap cannot be finer than the section count allows: with 3 sections and
    # 2 batches the best possible split is 2/1, i.e. 67%. Demanding 60% of a
    # 3-section warm-up would be demanding an impossible package, so the bound
    # is the stricter of the policy and what the granularity can express.
    n_sec = key["blind_id"].nunique()
    floor = -(-n_sec // 2) / n_sec
    limit = max(0.62, floor + 1e-9)
    check(f"no staining batch above {100 * limit:.0f}% of fields",
          shares.max() <= limit,
          "; ".join(f"{b} {100 * v:.1f}%" for b, v in shares.items())
          + (f" (granularity floor {100 * floor:.0f}%)" if floor > 0.62 else ""))
    check("a repeat never reuses its original's field id",
          key["field_id"].is_unique)
    dup_images = key.groupby("image")["blind_id"].nunique()
    check("no image is shared between two sections", (dup_images == 1).all())

    # ---- 3. the page matches the key -------------------------------------
    data = json.loads(re.search(
        r'<script type="application/json" id="data">(.*?)</script>',
        html, re.S).group(1))
    n_fields = sum(len(s["fields"]) for s in data)
    check("page section count matches key.csv",
          len(data) == key["blind_id"].nunique(),
          f"page {len(data)}, key {key['blind_id'].nunique()}")
    check("page field count matches key.csv", n_fields == len(key),
          f"page {n_fields}, key {len(key)}")

    referenced = {f[k] for s in data for f in s["fields"] for k in ("i", "c")}
    missing = [r for r in sorted(referenced) if not (pkg / r).exists()]
    check("every image the page references exists", not missing,
          f"{len(missing)} missing")
    on_disk = {p.relative_to(pkg).as_posix()
               for sub in ("images", "context") for p in (pkg / sub).glob("*.jpg")}
    orphans = on_disk - referenced
    check("no image ships that the page never shows", not orphans,
          f"{len(orphans)} orphaned files")

    # ---- 4. blinding -----------------------------------------------------
    # Nothing that identifies the slide, its cohort, its batch, its detector
    # score or which side of the split it is on may reach the browser.
    leaks = []
    for col, label in (("slide", "slide name"), ("cohort", "cohort"),
                       ("batch", "batch"), ("split", "split")):
        for value in {str(v) for v in key[col] if str(v).strip()}:
            if value in html:
                leaks.append(f"{label} {value!r}")
    for token in ("sample_type", "repeat", "uniform", "score", "MASH", "CCl4",
                  "chow"):
        if re.search(rf'"{token}"', html):
            leaks.append(f"key token {token!r}")
    check("no slide, cohort, batch, split or stratum reaches the page",
          not leaks, "; ".join(sorted(set(leaks))[:6]))
    check("no detector score reaches the page",
          not re.search(r'"score"', html))

    # ---- 5. the feature set actually in the shipped page -----------------
    js = extract_script(html)
    features = {
        "circle placement": "circles.push(",
        "drag to move": "mode==='move'",
        "corner handle resize": "mode==='resize'",
        "delete badge (x on every circle)": "data-x=",
        "keyboard delete": "e.key==='Delete'",
        "zoom on scroll": "'wheel'",
        "Y asks before an empty yes": "!circles.length && !armed",
        "back a field": "goBack",
        "section montage grading": "openGrade",
        "montage deduplicates repeats": "uniqueFields",
        "NASH-CRN 0/1/2": "setGrade('2')",
        "continuous file sync": "createWritable",
        "resume from a saved handle": "loadHandle",
        "checkpoints": "CHECKPOINTS",
        "carry on after a partial save": "carryOn",
        "per-field timing": "performance.now()",
    }
    for label, needle in features.items():
        check(f"page has: {label}", needle in js)
    check("high-contrast rings (white + dark halo)",
          "rgba(255,255,255,.95)" in html and "rgba(0,0,0,.45)" in html)

    # ---- 6. it works with no network -------------------------------------
    urls = re.findall(r'(?:href|src)="(https?://[^"]+)"', html)
    check("no external stylesheet, script or image", not urls,
          "; ".join(urls[:3]))
    check("no fetch/XHR to anywhere",
          not re.search(r"\b(fetch|XMLHttpRequest|WebSocket)\s*\(", js))

    # ---- 7. size ----------------------------------------------------------
    total = sum(f.stat().st_size for f in pkg.rglob("*") if f.is_file())
    # A ceiling on what is reasonable to hand someone, not a quality budget.
    # The full 9-batch study is ~120 sections of full-resolution fields plus
    # their context panels; it is meant to be large.
    check("package under 1 GB", total < 1e9, f"{total / 1e6:.0f} MB")

    # ---- 8. drive the shipped script -------------------------------------
    harness = ROOT / "tests/browser_full/harness.js"
    with tempfile.TemporaryDirectory() as td:
        shipped = Path(td) / "shipped.js"
        shipped.write_text(js)
        r = subprocess.run(["node", str(harness), str(shipped)],
                           capture_output=True, text=True)
        check("the SHIPPED script passes the full browser flow", r.returncode == 0,
              (r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else "")

    # ---- 9. the thing she double-clicks, from a fresh copy ----------------
    if not web:
        check("START_MAC.command serves the package from a fresh copy",
              *launcher_works(pkg))


def launcher_works(pkg: Path) -> tuple[bool, str]:
    """Copy the package somewhere new and run START_MAC.command for real.

    `open` is shimmed on PATH so the check does not hijack a browser window;
    everything else -- the cd, the port, the server -- is the script as shipped.
    """
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fresh = td / "Fresh Copy"        # a space in the path, as a Desktop has
        shutil.copytree(pkg, fresh)
        binf = td / "bin"
        binf.mkdir()
        (binf / "open").write_text('#!/bin/sh\necho "$@" > "$(dirname "$0")/opened.txt"\n')
        (binf / "open").chmod(0o755)
        env = {**os.environ, "PATH": f"{binf}:{os.environ['PATH']}"}
        proc = subprocess.Popen(["/bin/bash", str(fresh / "START_MAC.command")],
                                cwd=td, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                start_new_session=True)
        try:
            url, body = None, ""
            for _ in range(50):
                if (binf / "opened.txt").exists():
                    url = (binf / "opened.txt").read_text().strip()
                    break
                subprocess.run(["sleep", "0.2"])
            if not url:
                return False, "the launcher never opened a URL"
            for _ in range(25):
                r = subprocess.run(["curl", "-fsS", url], capture_output=True,
                                   text=True)
                if r.returncode == 0:
                    body = r.stdout
                    break
                subprocess.run(["sleep", "0.2"])
            if not body:
                return False, f"nothing served at {url}"
            if "Ballooning" not in body:
                return False, f"{url} served something unexpected"
            return True, f"{url} served the page"
        finally:
            os.killpg(os.getpgid(proc.pid), 15)
            proc.wait(timeout=10)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    verify(Path(argv[1]), Path(argv[2]))
    width = max(len(n) for n, _, _ in CHECKS)
    failed = 0
    for name, ok, detail in CHECKS:
        failed += not ok
        mark = "PASS" if ok else "FAIL"
        print(f"  {mark} {name:{width}}  {detail}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
