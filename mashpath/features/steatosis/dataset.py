"""Export a training-ready weakly supervised dataset.

Produces, per tile: the image, a consensus pseudo-label, and a per-pixel
confidence map. The confidence map is the point -- it lets a segmentation model
weight its loss by how certain each pseudo-label pixel is, rather than treating
programmatic labels as ground truth.

Two design decisions that matter more than they look:

**Splits are by slide, never by tile.** Adjacent tiles from one slide share
staining, scanner state, and often the same hepatocytes at their shared border.
Splitting by tile leaks all of that into validation and produces scores that
collapse on genuinely unseen slides.

**Fat-free tissue is included as negatives.** A model trained only on steatotic
tissue has never been shown liver without fat and will invent droplets on it.
The CCl4 cohort supplies real negatives with empty masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from ...config import MashConfig as Config
from .ensemble import EnsembleMember, consensus, default_ensemble
from ...core.slide import Slide
from ...core.tiling import iter_tiles
from ...core.tissue import detect_tissue
from ...core.viz import save_rgb


@dataclass
class SplitPlan:
    train: list[str]
    val: list[str]
    test: list[str]

    def of(self, slide: str) -> str:
        if slide in self.train:
            return "train"
        if slide in self.val:
            return "val"
        return "test"

    def to_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def plan_splits(
    slides: Sequence[str],
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 0,
    strata: dict[str, float] | None = None,
) -> SplitPlan:
    """Assign whole slides to train/val/test.

    With `strata` (slide -> severity), slides are sorted by severity and dealt
    so every split spans the full range. This matters at this sample size: a
    plain random split of 14 slides put all three fattiest slides in test
    (13.6-15.1%) and left train at 3.6-12.8%, so the model would have been
    evaluated entirely outside its training range.

    Without `strata` the split is random with a fixed seed.
    """
    ordered = sorted(slides)
    n = len(ordered)
    n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
    n_val = max(1, int(round(n * val_frac))) if n > 2 else 0
    n_train = n - n_test - n_val

    if strata is None:
        rng = np.random.default_rng(seed)
        shuffled = [ordered[int(i)] for i in rng.permutation(n)]
        return SplitPlan(
            train=sorted(shuffled[n_test + n_val :]),
            val=sorted(shuffled[n_test : n_test + n_val]),
            test=sorted(shuffled[:n_test]),
        )

    # Deal along the severity ordering, each slide going to whichever split is
    # furthest from filling its quota. That interleaves the splits through the
    # range instead of handing out contiguous blocks.
    quota = {"train": n_train, "val": n_val, "test": n_test}
    left = dict(quota)
    buckets: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for slide in sorted(ordered, key=lambda s: (strata.get(s, 0.0), s)):
        pick = max(
            (k for k in left if left[k] > 0),
            key=lambda k: (left[k] / quota[k], quota[k], k),
            default=None,
        )
        if pick is None:
            pick = "train"
        buckets[pick].append(slide)
        left[pick] = max(0, left[pick] - 1)
    return SplitPlan(
        train=sorted(buckets["train"]),
        val=sorted(buckets["val"]),
        test=sorted(buckets["test"]),
    )


def estimate_slide_fat(
    slide_path: str | Path,
    cfg: Config,
    members: Sequence[EnsembleMember],
    n_tiles: int = 8,
    seed: int = 0,
) -> float:
    """Rough slide-level fat fraction from a few tiles, for stratifying splits."""
    slide = Slide(str(slide_path), cfg.slide)
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        tiles = list(iter_tiles(slide, tissue, cfg.tiling))
        if not tiles:
            return 0.0
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(tiles), min(n_tiles, len(tiles)), replace=False)
        um2 = slide.um2_per_pixel(cfg.tiling.level)
        fat = tis = 0.0
        for i in idx:
            t = tiles[int(i)]
            rgb = t.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(t.x, t.y, t.size, t.size)
            r = consensus(rgb, tmask, members, um2)
            fat += float(r.mask_at(0.5).sum()) * um2
            tis += r.tissue_area_um2
        return fat / tis if tis else 0.0
    finally:
        slide.close()


def export_slide(
    slide_path: str | Path,
    cfg: Config,
    members: Sequence[EnsembleMember],
    out_dir: Path,
    split: str,
    cohort: str,
    max_tiles: int,
    seed: int,
    agree: float = 0.5,
    save_images: bool = True,
    extra: dict | None = None,
) -> list[dict]:
    """Export one slide's tiles with consensus labels and confidence maps.

    `split` may be "" to write flat into `out_dir` instead of into a
    train/val/test subdirectory -- see `build_flat_dataset` for why that is the
    preferred layout now. `extra` is merged into every manifest row, which is
    how batch and diet ride along without this function needing to know what
    they are.
    """
    slide = Slide(str(slide_path), cfg.slide)
    rows: list[dict] = []
    try:
        tissue = detect_tissue(slide, cfg.tissue)
        tiles = list(iter_tiles(slide, tissue, cfg.tiling))
        if not tiles:
            return []
        rng = np.random.default_rng(seed)
        if max_tiles and len(tiles) > max_tiles:
            idx = rng.choice(len(tiles), max_tiles, replace=False)
            tiles = [tiles[int(i)] for i in sorted(idx)]

        um2 = slide.um2_per_pixel(cfg.tiling.level)
        name = Path(slide_path).stem
        dest = out_dir / split if split else out_dir
        for sub in ("images", "labels", "confidence", "tissue"):
            (dest / sub).mkdir(parents=True, exist_ok=True)

        for t in tiles:
            rgb = t.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(t.x, t.y, t.size, t.size)
            r = consensus(rgb, tmask, members, um2)
            mask = r.mask_at(agree)
            conf = (r.confidence * 255).astype(np.uint8)
            stem = f"{name}__{t.name}"

            if save_images:
                save_rgb(dest / "images" / f"{stem}.png", rgb)
                m8 = (mask.astype(np.uint8) * 255)
                save_rgb(dest / "labels" / f"{stem}.png",
                         np.dstack([m8, m8, m8]))
                save_rgb(dest / "confidence" / f"{stem}.png",
                         np.dstack([conf, conf, conf]))
                # The tissue mask, because glass is white and a model given a
                # tile with background in it has every reason to call the
                # background fat. The teacher never could -- `extract_components`
                # does `white &= tissue_mask` before it measures anything -- so
                # without this the student is scored on a larger canvas than the
                # teacher was, and its fat fraction can exceed 1. 5.8% of tiles
                # in this build are under 90% tissue, so it is not a corner case.
                t8 = (tmask.astype(np.uint8) * 255)
                save_rgb(dest / "tissue" / f"{stem}.png",
                         np.dstack([t8, t8, t8]))

            st = r.agreement_stats(agree)
            rows.append({
                "id": stem,
                "split": split,
                "cohort": cohort,
                "slide": name,
                "tile_x": t.x,
                "tile_y": t.y,
                "tile_size": t.size,
                "mpp": slide.mpp_x,
                "tissue_fraction": round(t.tissue_fraction, 4),
                "fat_fraction": round(r.fat_fraction(agree), 6),
                "label_px": int(mask.sum()),
                "unanimous_px": int(st["unanimous_px"]),
                "any_px": int(st["any_px"]),
                "unanimous_of_any": round(st["unanimous_of_any"], 4),
                "contested_of_label": round(st["contested_of_consensus"], 4),
                "is_negative": cohort == "negative",
                **(extra or {}),
            })
        return rows
    finally:
        slide.close()


def build_dataset(
    cohorts: dict[str, Iterable[str]],
    cfg: Config,
    out_dir: str | Path,
    max_tiles_per_slide: int = 200,
    agree: float = 0.5,
    seed: int = 0,
    save_images: bool = True,
    stratify_by_fat: bool = True,
) -> pd.DataFrame:
    """Build the full dataset.

    `cohorts` maps a cohort label ("positive" / "negative") to slide paths.
    Splits are planned per cohort so both appear in train, val and test, and
    are stratified by estimated severity unless `stratify_by_fat` is off.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = default_ensemble(cfg.fat)

    all_rows: list[dict] = []
    plans: dict[str, dict[str, list[str]]] = {}
    failures: list[dict[str, str]] = []

    for cohort, paths in cohorts.items():
        paths = list(paths)

        # Drop unreadable slides BEFORE planning, so the recorded split never
        # claims a slide that produced no tiles. A previous run silently left
        # a truncated slide in splits.json while it was absent from the
        # manifest -- the two artifacts disagreed and nothing said why.
        usable: list[str] = []
        for p in paths:
            try:
                Slide(str(p), cfg.slide).close()
                usable.append(p)
            except Exception as exc:
                failures.append({
                    "slide": Path(p).stem, "cohort": cohort,
                    "stage": "open", "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  UNREADABLE {Path(p).stem}: {type(exc).__name__}", flush=True)

        strata = None
        if stratify_by_fat and len(usable) > 2:
            print(f"{cohort}: estimating severity for stratified split...", flush=True)
            strata = {
                Path(p).stem: estimate_slide_fat(p, cfg, members, seed=seed)
                for p in usable
            }

        stems = [Path(p).stem for p in usable]
        plan = plan_splits(stems, seed=seed, strata=strata)
        plans[cohort] = plan.to_dict()
        print(f"{cohort}: train={len(plan.train)} val={len(plan.val)} "
              f"test={len(plan.test)} slides", flush=True)

        for p in usable:
            stem = Path(p).stem
            split = plan.of(stem)
            try:
                rows = export_slide(
                    p, cfg, members, out_dir, split, cohort,
                    max_tiles_per_slide, seed, agree, save_images,
                )
                all_rows.extend(rows)
                print(f"  [{split:5}] {stem}: {len(rows)} tiles", flush=True)
            except Exception as exc:
                failures.append({
                    "slide": stem, "cohort": cohort,
                    "stage": "export", "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  SKIP {stem}: {type(exc).__name__}: {exc}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "splits.json").write_text(json.dumps(plans, indent=2))
    # Always written, empty list included, so its absence is never ambiguous.
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    if failures:
        print(f"\n{len(failures)} slide(s) excluded -- see failures.json", flush=True)
    cfg.save(out_dir / "config_used.yaml")
    (out_dir / "ensemble.json").write_text(
        json.dumps(
            [{"name": m.name,
              "white_threshold": m.cfg.white_threshold,
              "min_area_um2": m.cfg.min_area_um2,
              "max_eccentricity": m.cfg.max_eccentricity,
              "circularity_min": m.cfg.circularity_min,
              "solidity_min": m.cfg.solidity_min} for m in members],
            indent=2,
        )
    )
    return df


# ---------------------------------------------------------------------------
# The 9-batch build. No baked splits.
# ---------------------------------------------------------------------------

def resolve_batches(paths: Sequence[str | Path],
                    dripped: str | Path = "cluster/.dripped"
                    ) -> tuple[dict[str, str], list[str]]:
    """slide stem -> staining batch, and the stems that had to be guessed.

    Batch is not in the SVS. `cluster/.dripped` is the authoritative record;
    the containing folder is the fallback, and on the lab's OneDrive the two
    happen to agree because both are the staining-run folder. The fallback is
    reported rather than silent: a slide whose batch was guessed from a
    directory someone could rename is a slide whose held-out-batch result
    cannot be trusted, and that has to be visible in the manifest's provenance
    rather than discovered later.
    """
    from ...review.tileset import batch_index
    index = batch_index(dripped)
    out, guessed = {}, []
    for p in paths:
        stem = Path(p).stem
        if stem in index:
            out[stem] = index[stem]
        else:
            out[stem] = Path(p).parent.name
            guessed.append(stem)
    return out, guessed


def _finalize_rows(rows: list[dict]) -> pd.DataFrame:
    """Manifest frame from exported rows: drop `split`, derive `cohort`.

    Shared by the local loop and the cluster's finalize step, so the two cannot
    produce manifests that differ in a column.
    """
    df = pd.DataFrame(rows)
    # `split` is always "" here and would be an invitation to use it. The whole
    # point of this layout is that the split does not exist until training
    # time, so the column does not either.
    df = df.drop(columns=["split"], errors="ignore")
    return df


def export_one(slide_path: str | Path, cfg: Config, out_dir: str | Path,
               max_tiles_per_slide: int = 150, agree: float = 0.5,
               seed: int = 0, dripped: str | Path = "cluster/.dripped",
               reserved: Sequence[str] | None = None) -> pd.DataFrame:
    """Export ONE slide and write its manifest fragment. The cluster entry point.

    On Fir the whole-collection loop is the wrong shape: a slide read straight
    off Lustre costs ~82x what the same read costs on local disk (measured, see
    `cluster/ballooning_range.sbatch`), so 260 slides in one job would spend
    almost all of its wall clock waiting on per-operation latency. One slide per
    array task, staged to node-local NVMe by the sbatch first, turns that back
    into the 30 s per slide it is on a laptop.

    Fragments rather than one shared manifest because array tasks run
    concurrently: appending to a single CSV from 15 processes interleaves rows
    and occasionally truncates one. `finalize_flat_dataset` concatenates them.
    """
    from ...review.tileset import diet_from_name
    from ...train.splits import RESERVED_BATCHES

    out_dir = Path(out_dir)
    stem = Path(slide_path).stem
    reserved = tuple(RESERVED_BATCHES if reserved is None else reserved)
    batch_of, guessed = resolve_batches([slide_path], dripped)
    if batch_of[stem] in reserved:
        raise ValueError(
            f"{stem} is in reserved batch {batch_of[stem]!r} and must not be "
            f"exported into the training set. Export it separately, into its "
            f"own directory, with reserved=().")

    rows = export_slide(slide_path, cfg, default_ensemble(cfg.fat), out_dir, "",
                        "unknown", max_tiles_per_slide, seed, agree, True,
                        extra={"batch": batch_of[stem],
                               "diet": diet_from_name(stem),
                               "batch_guessed": bool(guessed)})
    parts = out_dir / "manifest_parts"
    parts.mkdir(parents=True, exist_ok=True)
    df = _finalize_rows(rows)
    df.to_csv(parts / f"{stem}.csv", index=False)
    return df


def finalize_flat_dataset(out_dir: str | Path, cfg: Config | None = None,
                          negative_below: float = 0.01) -> pd.DataFrame:
    """Concatenate the fragments an array run left, and write the manifest.

    Deliberately separate from the export so it can be rerun after a failed
    task is resubmitted, without re-exporting the 250 slides that worked.
    """
    out_dir = Path(out_dir)
    parts = sorted((out_dir / "manifest_parts").glob("*.csv"))
    if not parts:
        raise FileNotFoundError(f"no manifest fragments under {out_dir}/manifest_parts")
    df = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
    dupes = df["id"].duplicated().sum()
    if dupes:
        raise ValueError(
            f"{dupes} duplicate tile id(s) across fragments -- a slide was "
            f"exported by more than one task. Fix the slide list before "
            f"training; duplicate rows weight that animal double.")
    df = _assign_cohort(df, negative_below)
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "provenance.json").write_text(json.dumps({
        "layout": "flat -- split at training time by mashpath.train.splits",
        "assembled_from_fragments": len(parts),
        "slides": int(df["slide"].nunique()),
        "batches": sorted(df["batch"].unique().tolist()),
        "tiles": int(len(df)),
        "negative_below": negative_below,
    }, indent=2))
    if cfg is not None:
        cfg.save(out_dir / "config_used.yaml")
    # The batch list the training array is driven by, written here so nobody
    # has to derive it by hand and get one name wrong.
    (out_dir / "batches.txt").write_text(
        "\n".join(sorted(df["batch"].unique())) + "\n")
    print(f"{len(df)} tiles, {df['slide'].nunique()} slides, "
          f"{df['batch'].nunique()} batches -> {out_dir}", flush=True)
    return df


def _assign_cohort(df: pd.DataFrame, negative_below: float) -> pd.DataFrame:
    """Cohort from what was measured, not declared: six of the nine batches
    have no metadata. Per slide, so a later metadata key can correct it without
    re-exporting 21 GB."""
    if df.empty:
        return df
    per_slide = df.groupby("slide")["fat_fraction"].mean()
    df = df.copy()
    df["cohort"] = np.where(df["slide"].map(per_slide) < negative_below,
                            "negative", "positive")
    df["is_negative"] = df["cohort"] == "negative"
    return df


def _export_one(args) -> tuple[str, list[dict], dict | None]:
    """One slide, in a worker process. Rebuilds the ensemble locally.

    `default_ensemble` is cheap and rebuilding it here avoids shipping nine
    config objects through the pickle for every slide; more importantly it
    means the workers cannot be handed a members list that has drifted from
    the config they are also given, which would be invisible in the output.
    """
    (path, cfg, out_dir, max_tiles, seed, agree, save_images, extra) = args
    stem = Path(path).stem
    try:
        rows = export_slide(path, cfg, default_ensemble(cfg.fat), Path(out_dir),
                            "", "unknown", max_tiles, seed, agree, save_images,
                            extra=extra)
        return stem, rows, None
    except Exception as exc:
        return stem, [], {"slide": stem, "stage": "export",
                          "error": f"{type(exc).__name__}: {exc}"}


def build_flat_dataset(
    slides: Sequence[str | Path],
    cfg: Config,
    out_dir: str | Path,
    max_tiles_per_slide: int = 150,
    agree: float = 0.5,
    seed: int = 0,
    save_images: bool = True,
    workers: int = 1,
    dripped: str | Path = "cluster/.dripped",
    reserved: Sequence[str] | None = None,
    negative_below: float = 0.01,
) -> pd.DataFrame:
    """Export every slide flat, with batch in the manifest and NO split on disk.

    **Why no train/val/test directories.** `build_dataset` writes them, planned
    by slide and stratified by severity, and that layout is the mistake
    `docs/BATCH_EFFECTS.md` §2 is about: it makes a slide-level split look like
    the model's split, and a slide-level split tests generalisation to a new
    animal rather than to a new stain. Two slides from one batch were cut on
    one day by one person from one reagent lot; holding one out proves nothing
    about the next staining run. With batch on every row, `mashpath.train.splits`
    does the folding at training time, where it can refuse a tile-level split
    outright and enforce the reserved batches. A directory called `train/`
    can do neither.

    **Reserved batches are not exported at all.** `splits.py` already prevents
    them reaching a training fold; leaving them out of the dataset as well means
    a training script that never imports `splits.py` still cannot touch them.
    The one measurement in this project that is not the detector grading its own
    homework is worth two locks.

    `negative_below` labels a slide `negative` when its exported tiles average
    less than this fat fraction. Derived rather than declared, because 6 of the
    9 batches have no cohort metadata -- and it is recorded per slide in the
    manifest so a later metadata key can overwrite it without a re-export.
    """
    from ...review.tileset import diet_from_name
    from ...train.splits import RESERVED_BATCHES

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    members = default_ensemble(cfg.fat)
    reserved = tuple(RESERVED_BATCHES if reserved is None else reserved)

    # One slide, one export. Tile ids are `<stem>__<tile>`, so the same slide
    # reached by two paths writes the same filenames twice and lands twice in
    # the manifest -- the second overwriting the first, the duplicate rows
    # silently weighting that animal double in training. Not hypothetical:
    # R26-122-1_HE_70.svs sits both at the top of /scratch/$USER/slides and
    # inside its batch folder, so the obvious glob over the cluster's slide
    # directory returns it twice.
    paths, seen, dupes = [], set(), []
    for q in (Path(x) for x in slides):
        if q.stem in seen:
            dupes.append(str(q))
            continue
        seen.add(q.stem)
        paths.append(q)
    if dupes:
        print(f"{len(dupes)} duplicate slide path(s) ignored, first: {dupes[0]}",
              flush=True)

    batch_of, guessed = resolve_batches(paths, dripped)

    held = [p for p in paths if batch_of[p.stem] in reserved]
    paths = [p for p in paths if batch_of[p.stem] not in reserved]
    if held:
        print(f"reserved: {len(held)} slide(s) from {sorted(set(reserved))} "
              f"excluded from the export entirely", flush=True)
    if guessed:
        print(f"batch guessed from the folder name for {len(guessed)} slide(s)",
              flush=True)

    # A batch of one is almost always a path that did not resolve rather than a
    # staining run with one slide in it -- `batch_index` recognises "no batch
    # folder" by the literal directory name `slides`, so a slide sitting at the
    # top of a differently-named slide root takes that root's name as its batch.
    # It would then become a leave-one-batch-out fold of one slide and report a
    # number that means nothing. Reported here rather than raised: it is a
    # smell, not always an error, and a build that refuses to run is worse than
    # one that says what it found.
    from collections import Counter
    sizes = Counter(batch_of[q.stem] for q in paths)
    singles = sorted(b for b, n in sizes.items() if n == 1)
    for b in singles:
        print(f"WARNING: batch {b!r} has ONE slide. Check it is a real staining "
              f"run and not an unresolved path -- as a held-out fold it would "
              f"report a meaningless number.", flush=True)

    jobs = [(str(p), cfg, str(out_dir), max_tiles_per_slide, seed, agree,
             save_images,
             {"batch": batch_of[p.stem], "diet": diet_from_name(p.stem),
              "batch_guessed": p.stem in guessed})
            for p in paths]

    all_rows: list[dict] = []
    failures: list[dict[str, str]] = []

    def _record(i: int, stem: str, rows: list[dict], err: dict | None) -> None:
        if err:
            err["batch"] = batch_of[stem]
            failures.append(err)
            print(f"  SKIP {stem}: {err['error']}", flush=True)
            return
        all_rows.extend(rows)
        fat = float(np.mean([r["fat_fraction"] for r in rows])) if rows else 0.0
        print(f"  [{i}/{len(jobs)}] {stem}: {len(rows)} tiles, "
              f"mean fat {fat * 100:.2f}%", flush=True)

    if workers > 1:
        # Each slide opens its own reader and writes filenames prefixed by its
        # own stem, so slides never contend for a file. Chunked at one job per
        # task because slides differ several-fold in tile count and a static
        # chunk would leave one worker holding the big ones.
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, (stem, rows, err) in enumerate(
                    pool.map(_export_one, jobs, chunksize=1), 1):
                _record(i, stem, rows, err)
    else:
        for i, job in enumerate(jobs, 1):
            _record(i, *_export_one(job))

    for r in all_rows:
        pass
    df = _finalize_rows(all_rows)
    df = _assign_cohort(df, negative_below)
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "provenance.json").write_text(json.dumps({
        "layout": "flat -- split at training time by mashpath.train.splits",
        "slides_exported": int(df["slide"].nunique()) if not df.empty else 0,
        "batches": sorted(df["batch"].unique().tolist()) if not df.empty else [],
        "reserved_excluded": sorted(set(reserved)),
        "reserved_slides_excluded": sorted(p.stem for p in held),
        "batch_guessed_from_folder": sorted(guessed),
        "duplicate_paths_ignored": sorted(dupes),
        "single_slide_batches": singles,
        "max_tiles_per_slide": max_tiles_per_slide,
        "agree": agree, "seed": seed,
        "negative_below": negative_below,
    }, indent=2))
    cfg.save(out_dir / "config_used.yaml")
    (out_dir / "ensemble.json").write_text(json.dumps(
        [{"name": m.name, "white_threshold": m.cfg.white_threshold,
          "min_area_um2": m.cfg.min_area_um2,
          "max_eccentricity": m.cfg.max_eccentricity,
          "circularity_min": m.cfg.circularity_min,
          "solidity_min": m.cfg.solidity_min} for m in members], indent=2))
    if not df.empty:
        print(f"\n{len(df)} tiles, {df['slide'].nunique()} slides, "
              f"{df['batch'].nunique()} batches -> {out_dir}", flush=True)
    return df
