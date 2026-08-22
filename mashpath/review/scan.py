"""Run a candidate detector over a slide and build its review package.

The same tissue -> tiles -> detect -> export path serves ballooning and
inflammation, so it lives here once rather than being copied into each feature.
A feature supplies only `detect_candidates`; everything around it is shared.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from ..core.io import SlideOutputs
from ..core.slide import Slide
from ..core.tiling import count_candidate_tiles, iter_tiles
from ..core.tissue import detect_tissue
from .candidates import Candidate, CandidateSet
from .export import export_candidates

Detector = Callable[..., Sequence[Candidate]]


def scan_slide(
    cfg: Any,
    feature: str,
    detector: Detector,
    feature_cfg: Any,
    plan_only: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tissue, tiling, detection, review export -- for any candidate feature.

    `plan_only` stops after the tile plan, which exercises everything except
    the detector itself. That is what makes the scaffolding testable while
    ballooning and inflammation are still stubs.
    """
    slide = Slide(cfg.slide_path, cfg.slide)
    try:
        if verbose:
            print(slide.describe())
            print()

        t0 = time.time()
        tissue = detect_tissue(slide, cfg.tissue)
        if verbose:
            print(
                f"tissue: {tissue.area_mm2:.2f} mm^2, "
                f"{tissue.coverage*100:.1f}% coverage, "
                f"{tissue.component_count} component(s) [{time.time()-t0:.1f}s]"
            )

        n_candidates = count_candidate_tiles(slide, tissue, cfg.tiling)
        n_process = min(cfg.limit, n_candidates) if cfg.limit else n_candidates
        if verbose:
            print(
                f"tiles:  {n_candidates} pass tissue_fraction >= "
                f"{cfg.tiling.min_tissue_fraction}"
                + (f"; --limit {cfg.limit} -> processing {n_process}"
                   if cfg.limit else "")
            )

        summary: dict[str, Any] = {
            "slide": slide.name,
            "feature": feature,
            "tiles_candidate": n_candidates,
            "tiles_processed": 0,
            "candidates": 0,
        }
        if plan_only:
            if verbose:
                print(f"\n  --plan: stopping before {feature} detection")
            return summary

        um2_px = slide.um2_per_pixel(cfg.tiling.level)
        found = CandidateSet(slide_id=slide.name, feature=feature)
        n_tiles = 0
        for tile in iter_tiles(slide, tissue, cfg.tiling, limit=cfg.limit):
            rgb = tile.read(slide, cfg.tiling.level)
            tmask = tissue.tile_mask(tile.x, tile.y, tile.size, tile.size)
            found.extend(
                detector(
                    rgb, tmask, feature_cfg, um2_px,
                    slide_id=slide.name, tile_x=tile.x, tile_y=tile.y,
                )
            )
            n_tiles += 1

        out = SlideOutputs.for_slide(cfg.output_dir, slide.name).mkdirs(feature)
        out.write_table(found.to_rows(), feature, "candidates")
        manifest = export_candidates(
            slide, found.candidates, out, feature, cfg.review, cfg.tiling.level
        )
        out.write_config(cfg)

        summary.update(
            tiles_processed=n_tiles,
            candidates=len(found),
            manifest=str(manifest),
            seconds=time.time() - t0,
        )
        if verbose:
            print(
                f"\n  {len(found)} {feature} candidate(s) from {n_tiles} tiles"
                f"\n  review package -> {manifest.parent}"
            )
        return summary
    finally:
        slide.close()
