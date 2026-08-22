"""End-to-end: slides in, a folder you can send to a pathologist out.

One command, so that when more slides land the only thing that changes is how
many directories are passed in. Everything downstream -- tissue detection, the
uniform draw, rendering, the page, the key -- is derived.

Detector scores are read where they exist but steer nothing: the draw is 100%
uniform. See the ENRICHMENT note in `fullset.py` for why.

The warm-up package is built from the SAME frame in the SAME call as the main
study, and that is not a convenience. Two separate invocations would each draw
from the whole frame and would overlap; building both here lets the warm-up
slides be held out of the main study by construction rather than by a check
someone has to remember to run.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from ..config import MashConfig
from . import fullset
from .tileset import TilePlan, build_frame, discover_sources, is_test_slide

WARMUP_SLIDES = 3
WARMUP_FIELDS_PER_SLIDE = 15      # ~51 presentations, ~13 min at 13 s/field


def build_frames(slide_dirs, score_dirs=(), config=("configs/core.yaml",),
                 plan_yaml="configs/ballooning_tiles.yaml",
                 out_csv=None, verbose=True, append=False) -> pd.DataFrame:
    """Tissue frames for the slides in `slide_dirs`.

    `append` accumulates into `out_csv` across a wave fetch. Without it, wave
    two overwrote study_frame.csv with only wave two's slides, and every
    downstream decision that is supposed to see the whole collection -- the
    batch-share cap, the time budget, which slides are dropped -- was then made
    on eight slides at a time.
    """
    cfg = MashConfig.from_yaml(list(config))
    plan = TilePlan.from_yaml(plan_yaml)
    srcs = discover_sources(list(slide_dirs), list(score_dirs))
    frames, failed = [], []
    for s in srcs:
        try:
            frames.append(build_frame(s, cfg, plan, verbose=False))
            if verbose:
                mark = "scored" if s.has_score else "uniform only"
                print(f"  {s.name:26s} {len(frames[-1]):5d} tiles   {mark}")
        except Exception as exc:
            failed.append((s.name, f"{type(exc).__name__}: {exc}"))
    if not frames:
        raise RuntimeError("no slide produced a frame")
    frame = pd.concat(frames, ignore_index=True)
    frame["split"] = ["test" if is_test_slide(x, plan) else "train"
                      for x in frame["slide"]]
    if verbose:
        print(f"\nframe: {len(frame)} tiles across {frame.slide.nunique()} slides")
        for n, why in failed:
            print(f"  FAILED {n}: {why}")
    if out_csv:
        out = Path(out_csv)
        if append and out.exists():
            prev = pd.read_csv(out)
            prev = prev[~prev["slide"].isin(set(frame["slide"]))]
            merged = pd.concat([prev, frame], ignore_index=True)
            merged.to_csv(out, index=False)
            if verbose:
                print(f"  accumulated: {merged.slide.nunique()} slides, "
                      f"{len(merged)} tiles in {out}")
        else:
            frame.to_csv(out, index=False)
    frame.attrs["failed"] = failed
    return frame


def _summary(nsec: int, nfield: int, per_section: list[int]) -> tuple[str, float]:
    hours = (nfield * fullset.FIELD_SECONDS
             + nsec * fullset.GRADE_SECONDS) / 3600.0
    lo, hi = min(per_section), max(per_section)
    span = f"{lo}" if lo == hi else f"{lo}&ndash;{hi}"
    if hours < 1:
        length = f"about {round(hours * 60 / 5) * 5} minutes"
    else:
        length = f"about {hours:.0f}&nbsp;hours"
    return (f"{nsec} sections &middot; {span} fields each &middot; {length} "
            f"to do all of it, in as many sittings as you like &mdash; "
            f"there is no need to finish"), hours


def assemble(package_dir, out_dir, app_html, app_js, round_name,
             verbose=True, title=None, order_note=None, image_root=None) -> Path:
    """Copy images and write the page. The page carries no slide identity."""
    package_dir, out_dir = Path(package_dir), Path(out_dir)
    image_root = Path(image_root) if image_root else package_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    key = pd.read_csv(package_dir / "key.csv")

    # Copy the files the key NAMES, not the directory it lives in. The work dir
    # accumulates renders from every draw ever made against it -- a re-draw
    # picks different tiles and leaves the old ones behind -- and copying the
    # directory shipped all of them: hundreds of MB of images the page never
    # references, in a package whose size is a thing she has to download.
    for sub in ("images", "context"):
        dst = out_dir / sub
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
    for col in ("image", "context"):
        for rel in sorted(set(key[col])):
            src = image_root / rel
            if not src.exists():
                raise FileNotFoundError(f"{src} is named in key.csv but missing")
            shutil.copy2(src, out_dir / rel)
    if "enriched" in set(key["sample_type"]):
        raise RuntimeError(
            f"{package_dir}/key.csv contains `enriched` rows. That stratum was "
            "removed because the detector score is anti-correlated with the "
            "cohort; a package carrying it was built from stale code. Rebuild.")

    sections = []
    for sid, grp in sorted(key.groupby("blind_id"),
                           key=lambda kv: kv[1].present_order.iloc[0]):
        g = grp.sort_values("panel")
        sections.append({"id": sid,
                         "fields": [{"id": r.field_id, "i": r.image, "c": r.context}
                                    for r in g.itertuples()]})

    nsec = len(sections)
    nfield = sum(len(s["fields"]) for s in sections)
    summary, hours = _summary(nsec, nfield, [len(s["fields"]) for s in sections])

    shell = Path(app_html).read_text()
    if "{{SUMMARY}}" not in shell:
        raise RuntimeError(f"{app_html} has no {{{{SUMMARY}}}} token — the page "
                           "would ship whatever the template last said.")
    shell = shell.replace("{{SUMMARY}}", summary)
    if title:
        shell = shell.replace("<title>Ballooning Annotation Study</title>",
                              f"<title>{title}</title>")
        shell = shell.replace("<h1>Ballooning Annotation Study</h1>",
                              f"<h1>{title}</h1>")
    js = Path(app_js).read_text()
    blob = json.dumps(sections, separators=(",", ":")).replace("</", "<\\/")
    warn = ("\nif(!storageOK){const w=document.getElementById('storeWarn');"
            "w.style.display='';w.textContent='This browser will not remember your "
            "progress between visits. Your results file is still written after every "
            "answer, so nothing is lost.';}\n")
    body = (f'\n<script type="application/json" id="data">{blob}</script>\n'
            f'<script>\nconst ROUND="{round_name}";\n{js}\n{warn}\n</script>\n')
    cut = shell.index("</style>") + len("</style>")
    (out_dir / "index.html").write_text(
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + shell[:cut] + "\n</head>\n<body>\n" + shell[cut:] + body
        + "</body>\n</html>\n")

    # The launchers matter more than they look: the File System Access API that
    # streams her answers to a file needs a real origin, and a page opened from
    # disk does not have one. These serve the folder on localhost instead.
    for extra in ("START_MAC.command", "START_WINDOWS.bat"):
        src = APP / extra
        if src.exists():
            shutil.copy2(src, out_dir / extra)
            if extra.endswith(".command"):
                (out_dir / extra).chmod(0o755)
    readme = (APP / "README.txt").read_text()
    # Unescape ALL entities, not a hand-kept list of three. `&mdash;` was added
    # to the summary later and went out raw in the plain-text README, which is
    # what a hardcoded list of entities always eventually does.
    import html as _html
    plain = _html.unescape(summary)
    for fancy, ascii_ in (("\u00b7", "-"), ("\u2013", "-"), ("\u2014", "--"),
                          ("\u00a0", " ")):
        plain = plain.replace(fancy, ascii_)
    if any(c in plain for c in "&<>"):
        raise RuntimeError(f"unconverted markup in the README summary: {plain!r}")
    name = (title or "Ballooning Annotation Study").upper()
    readme = readme.replace("{{TITLE}}", name).replace("{{RULE}}", "=" * len(name))
    readme = readme.replace("{{SUMMARY}}", plain)
    readme = readme.replace("{{ORDER}}", order_note or SOLO_NOTE)
    (out_dir / "README.txt").write_text(readme)

    if verbose:
        mb = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
        print(f"\npackage -> {out_dir}")
        print(f"  {nsec} sections, {nfield} fields, ~{hours:.1f} h of reviewer time")
        print(f"  {mb:.0f} MB")
    return out_dir


SOLO_NOTE = """BEFORE YOU COMMIT AN EVENING TO IT
  Please do two or three sections and then stop for a moment. After the
  second one the tool asks whether this is working -- whether you can
  judge ballooning from these images, and whether the circling is getting
  in the way. Answering that box, and emailing us the file as it stands,
  costs you fifteen minutes and is the cheapest chance to change anything.
  Everything you have done so far still counts.

  There is no deadline and no need to finish. The sections are ordered so
  that however far you get is a fair sample of the whole set -- every
  staining batch appears in the first fifteen."""

FIRST_NOTE = """DO THIS ONE FIRST
  This is the short warm-up. It is a real part of the study -- your answers
  count -- but it is deliberately small so that anything wrong with the
  images or the tool turns up in fifteen minutes rather than four hours.
  When you finish, send the file back before starting the main study."""

SECOND_NOTE = """BEFORE YOU START
  Please do the short Ballooning_Warmup package first and send that file
  back. It takes about fifteen minutes and it is what tells us the images
  and the tool are right before you commit a longer stretch to this one."""

APP = Path(__file__).parent / "app"


def build_study(slide_dirs, out_dir, work_dir="review_sets/ballooning_full",
                score_dirs=("bal_chunks", "bal_local"),
                app_html=None, app_js=None, round_name="ballooning_full_v1",
                append: bool = False, verbose: bool = True,
                hours: float | None = None,
                warmup_dir=None, warmup_work_dir="review_sets/ballooning_warmup",
                warmup_slides: int | None = None,
                reuse_frame: bool = False, stage: str = "both",
                fields_per_section: int | None = None) -> Path | None:
    """Build the main study and, unless `warmup_dir` is None, the warm-up too.

    `stage` exists for the wave fetch, where 65 GB of slides pass through a few
    GB of disk and no two waves are ever on the machine together:

      render    for each wave -- accumulate the frame, and render the MOST any
                allocation could ask of these slides (`MAX_PER_SECTION`), while
                their pixels are still here.
      assemble  once, at the end -- decide the allocation over EVERY slide and
                write the packages from the rendered images. Opens no slide.
      both      the default, for when every slide is already on disk.

    The split matters because the batch-share cap and the time budget are
    properties of the whole collection. Deciding them per wave decides them on
    eight slides at a time, which is not the same question.
    """
    # Resolved at call time, like the constants in fullset -- a default
    # argument freezes the module value at import and quietly ignores anyone
    # who changes it afterwards.
    hours = fullset.BUDGET_HOURS if hours is None else hours
    warmup_slides = WARMUP_SLIDES if warmup_slides is None else warmup_slides
    if stage not in ("render", "assemble", "both"):
        raise ValueError(f"unknown stage {stage!r}; "
                         f"expected render, assemble or both")
    app_html = app_html or APP / "study.html"
    app_js = app_js or APP / "study.js"
    frame_csv = Path(work_dir).parent / "study_frame.csv.gz"
    Path(work_dir).parent.mkdir(parents=True, exist_ok=True)

    if stage == "assemble":
        if not Path(frame_csv).exists():
            raise FileNotFoundError(f"{frame_csv} does not exist; run the render "
                                    f"stage over the slides first")
        frame = pd.read_csv(frame_csv)
        if verbose:
            print(f"frame: {len(frame)} tiles across {frame.slide.nunique()} "
                  f"slides, accumulated over every wave")
    elif reuse_frame and Path(frame_csv).exists():
        frame = pd.read_csv(frame_csv)
        if verbose:
            print(f"frame: reusing {frame_csv} "
                  f"({len(frame)} tiles, {frame.slide.nunique()} slides)")
    else:
        if verbose:
            print("frames:")
        frame = build_frames(slide_dirs, score_dirs, out_csv=frame_csv,
                             verbose=verbose, append=(stage == "render"))

    if stage == "render":
        # Render the ceiling, not the allocation. `_spread` is prefix-stable --
        # the first k of a 24-tile draw ARE the k-tile draw at the same seed --
        # so whatever the final allocation turns out to be, its fields are a
        # prefix of what is rendered here and no slide has to come back.
        wave = sorted(frame["slide"].unique())
        if verbose:
            print(f"\nrendering {len(wave)} slide(s) at "
                  f"{fullset.MAX_PER_SECTION} fields each (the ceiling):")
        fullset.build(frame_csv, list(slide_dirs), work_dir, verbose=verbose,
                      append=True, only_slides=wave,
                      per_slide={s: fullset.MAX_PER_SECTION for s in wave})
        return None

    frame = refresh_derived(frame, frame_csv if stage == "assemble" else None,
                            verbose=verbose)

    # Two passes. The first is thrown away except for its discard list: it says
    # which slides the batch cap is going to drop anyway, and the warm-up is
    # taken from there so that reserving it costs the main study almost nothing.
    if warmup_dir and warmup_slides:
        _, first = fullset.plan_allocation(
            frame[["slide", "batch"]], hours=hours,
            **(dict(min_per=fields_per_section, max_per=fields_per_section)
               if fields_per_section else {}))
        held = fullset.choose_warmup_slides(frame, warmup_slides,
                                            prefer=first["dropped"])
    else:
        held = []
    if held and verbose:
        free = sum(s in set(first["dropped"]) for s in held)
        print(f"\nheld out for the warm-up: {', '.join(held)}"
              f"  ({free}/{len(held)} were being dropped anyway)")

    main_slides = frame[~frame["slide"].isin(held)]
    depth = dict(min_per=fields_per_section, max_per=fields_per_section) \
        if fields_per_section else {}
    per_slide, alloc = fullset.plan_allocation(
        main_slides[["slide", "batch"]], hours=hours, **depth)
    if verbose:
        print(f"\nallocation: {alloc['sections']} sections x "
              f"{alloc['fields_per_section']} fields = "
              f"{alloc['presentations']} presentations, ~{alloc['hours']:.2f} h "
              f"at {alloc['field_seconds']:.0f} s/field "
              + (f"(budget {alloc['budget_hours']:.1f} h)"
                 if alloc['budget_hours'] else "(no time cap: every slide)"))
        for b, n in sorted(alloc["batch_slides"].items()):
            print(f"  {b:34s} {n:3d} slides  {100 * alloc['batch_share'][b]:5.1f}% "
                  f"of fields")
        print(f"  largest batch share {100 * alloc['share_before']:.1f}% -> "
              f"{100 * max(alloc['batch_share'].values()):.1f}% after dropping "
              f"{alloc['dropped_for_share']} slides")
        if alloc["dropped_for_budget"]:
            print(f"  dropped for budget: {alloc['dropped_for_budget']} slides")

    if verbose:
        print("\nrendering main study:")
    fullset.build(frame_csv, list(slide_dirs), work_dir, verbose=False,
                  append=append, per_slide=per_slide, exclude_slides=held)
    assemble(work_dir, out_dir, app_html, app_js, round_name, verbose,
             title="Ballooning Annotation Study",
             order_note=SECOND_NOTE if warmup_dir else None)

    if warmup_dir and held:
        if verbose:
            print("\nrendering warm-up:")
        fullset.build(frame_csv, list(slide_dirs), warmup_work_dir, verbose=False,
                      per_slide={s: WARMUP_FIELDS_PER_SLIDE for s in held},
                      only_slides=held, image_root=work_dir)
        assemble(warmup_work_dir, warmup_dir, app_html, app_js,
                 round_name + "_warmup", verbose,
                 title="Ballooning Warm-up", order_note=FIRST_NOTE,
                 image_root=work_dir)
        check_disjoint(work_dir, warmup_work_dir)
    return out_dir


def refresh_derived(frame: pd.DataFrame, write_to=None, verbose=True) -> pd.DataFrame:
    """Recompute cohort and diet from the slide name.

    Both are pure functions of the name, and the frame accumulates over waves
    that may span a change to either function -- as happened when accession
    parsing was fixed mid-fetch and every underscore-separated name had been
    resolving to cohort "unknown". A derived column that can go stale in a file
    should be re-derived where it is used, not trusted because it is written
    down.
    """
    from .tileset import COHORT_BY_PREFIX, _prefix, diet_from_name

    want_cohort = frame["slide"].map(
        lambda n: COHORT_BY_PREFIX.get(_prefix(n), "unknown"))
    want_diet = frame["slide"].map(diet_from_name)
    changed = set()
    for col, want in (("cohort", want_cohort), ("diet", want_diet)):
        if col in frame.columns:
            diff = frame[col].astype(str) != want.astype(str)
            if diff.any():
                changed |= set(frame.loc[diff, "slide"].unique())
        frame[col] = want
    if changed and verbose:
        print(f"  refreshed cohort/diet for {len(changed)} slide(s): "
              f"{', '.join(sorted(changed)[:4])}"
              f"{' ...' if len(changed) > 4 else ''}")
    if changed and write_to:
        frame.to_csv(write_to, index=False)
    return frame


def check_disjoint(a, b) -> None:
    """Fail loudly if the two packages share a field. Never assume it."""
    sa, sb = fullset.key_slides(a), fullset.key_slides(b)
    if sa & sb:
        raise RuntimeError(f"packages share slides: {sorted(sa & sb)}")
    ta, tb = fullset.key_tiles(a), fullset.key_tiles(b)
    if ta & tb:
        raise RuntimeError(f"packages share {len(ta & tb)} fields")
