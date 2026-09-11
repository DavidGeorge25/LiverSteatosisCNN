"""Render a precomputed slide into a static site — no server, no Python, no GPU.

The live viewer needs Fir: it region-reads a 125 GB slide collection through
OpenSlide and keeps a checkpoint in memory. None of that can go on a static
host, so this pre-renders what the browser actually asks for -- the DeepZoom
tiles of the tissue and of the mask -- into plain files that any static host
will serve. Pan, zoom, the overlay toggle and the opacity slider all keep
working, because OpenSeadragon was only ever fetching tiles over HTTP anyway.

WHAT IS LOST, AND IT IS ONLY ONE THING: the live spot check. Everything else,
including the census percentage, was already computed ahead of time -- the
server was reading it out of `meta.json`, not deriving it per request.

TILES ARE SKIPPED WHERE THERE IS NO TISSUE. A whole-slide pyramid is mostly
glass, and a 404 is how OpenSeadragon already expects to find the edge of an
image. Skipping them takes a fatty slide from roughly 3,400 full-resolution
tiles to about 1,500 and is the difference between a repo you can push and one
you cannot.

DEPTH IS CAPPED, DELIBERATELY. `--min-downsample 2` stops at half resolution,
which is ~1 um/px -- a 20 um macrovesicular droplet is still 20 px across and
the overlay reads clearly, at a quarter of the bytes of full resolution. Raise
it to 1 for a true full-resolution export if the host can take it.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..core.slide import Slide
from ..core.tissue import detect_tissue
from .dzi import DeepZoom, MaskTileSource, SlideTileSource

STATIC = Path(__file__).parent / "static"


def _jpeg(rgb: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def _png(arr: np.ndarray) -> bytes:
    """Overlay tile, compressed hard.

    OpenCV's default PNG compression level is 1, not 9 -- tuned for writing
    frames in a loop, not for bytes that are about to be committed to a repo
    and served forever. These tiles are the ideal case for the slow setting:
    three of the four channels are a constant colour and the fourth is a
    near-binary mask, so the extra effort costs a few milliseconds once and is
    lossless.
    """
    ok, buf = cv2.imencode(".png", arr, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def slug(name: str) -> str:
    """A URL-safe directory name.

    Slide names in this cohort carry spaces and ampersands -- "R22-354_25_
    ACLY656 NASH 6" -- and a directory named that produces tile URLs full of
    %20. Static hosts mostly cope, but "mostly" is not a property you want to
    discover from a PI saying the link is broken. The human-readable name
    stays in slides.json for the picker.
    """
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in "-_") else "-")
    s = "".join(out)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "slide"


def export_slide(artifact: str | Path, out_dir: str | Path,
                 cfg: Config | None = None, quality: int = 72,
                 min_downsample: int = 2, verbose: bool = True) -> dict:
    """Write one precomputed slide's tiles under `out_dir/<slug>/`."""
    cfg = cfg or Config()
    artifact = Path(artifact)
    meta = json.loads((artifact / "meta.json").read_text())
    name = meta["slide"]
    dirname = slug(name)
    dest = Path(out_dir) / dirname
    (dest / "slide_files").mkdir(parents=True, exist_ok=True)
    (dest / "overlay_files").mkdir(parents=True, exist_ok=True)

    sl = Slide(meta["slide_path"], cfg.slide)
    tissue = detect_tissue(sl, cfg.tissue)
    src = SlideTileSource(sl)
    mask = MaskTileSource(artifact, meta["width"], meta["height"],
                          mask_tile=int(meta.get("mask_tile", 512)))
    dz: DeepZoom = src.dz

    # Deepest level to export. `scale` is level-0 pixels per pixel, so a
    # min_downsample of 2 keeps every level whose scale is >= 2.
    deepest = dz.max_level
    while deepest > 0 and dz.scale(deepest) < min_downsample:
        deepest -= 1

    n_tiles = n_skipped = n_bytes = 0
    for level in range(deepest + 1):
        cols, rows = dz.tile_counts(level)
        (dest / "slide_files" / str(level)).mkdir(parents=True, exist_ok=True)
        (dest / "overlay_files" / str(level)).mkdir(parents=True, exist_ok=True)
        for row in range(rows):
            for col in range(cols):
                x0, y0, w0, h0, tw, th = dz.tile_box(level, col, row)
                # Skip pure background. `tissue_fraction` answers from the
                # level-2 mask, so this costs no level-0 read at all.
                if tissue.tissue_fraction(x0, y0, w0, h0) <= 0.0:
                    n_skipped += 1
                    continue
                jpg = _jpeg(src.tile(level, col, row), quality)
                (dest / "slide_files" / str(level) / f"{col}_{row}.jpg").write_bytes(jpg)
                n_bytes += len(jpg)
                n_tiles += 1

                g = mask.tile_gray(level, col, row)
                if not g.any():
                    continue          # no fat here: 404 and the layer is clear
                png = _png(mask.tile_rgba(level, col, row))
                (dest / "overlay_files" / str(level) / f"{col}_{row}.png").write_bytes(png)
                n_bytes += len(png)
        if verbose:
            print(f"  level {level:>2} ({dz.scale(level):>6.0f}x): "
                  f"{cols}x{rows} grid, {n_tiles} tiles so far", flush=True)

    # DZI descriptors. `Format` differs but the geometry is identical, which is
    # what keeps the two layers registered.
    (dest / "slide.dzi").write_text(dz.dzi_xml("jpg"))
    (dest / "overlay.dzi").write_text(dz.dzi_xml("png"))

    # The stats panel reads this. Trimmed to what the page shows, and the
    # slide's path on the cluster is dropped -- it means nothing off Fir and
    # there is no reason to publish a filesystem layout.
    keep = {k: meta[k] for k in (
        "slide", "width", "height", "mpp", "fat_percent", "fat_px", "tissue_px",
        "tissue_mm2", "tiles", "coverage", "threshold", "mask_tile",
        "training_mpp", "resampled_to_training_mpp", "droplets", "certainty",
        "distribution", "seconds", "ms_per_tile", "validated_against_reference",
        "validation_note") if k in meta}
    keep["model"] = {k: meta.get("model", {}).get(k)
                     for k in ("fold", "held_out_batch", "stain_normalize")}
    keep["max_exported_level"] = deepest
    keep["max_exported_downsample"] = dz.scale(deepest)
    (dest / "meta.json").write_text(json.dumps(keep, indent=2, default=float))
    sl.close()

    if verbose:
        print(f"  {name}: {n_tiles} tiles, {n_skipped} skipped as background, "
              f"{n_bytes / 1e6:.1f} MB", flush=True)
    return {"name": name, "dir": dirname, "tiles": n_tiles,
            "skipped": n_skipped, "bytes": n_bytes,
            "fat_percent": meta["fat_percent"],
            "deepest_level": deepest, "deepest_downsample": dz.scale(deepest)}


def export_site(artifacts: list[str | Path], out_dir: str | Path,
                cfg: Config | None = None, quality: int = 72,
                min_downsample: int = 2, title: str = "",
                verbose: bool = True) -> dict:
    """Export several slides plus the page that reads them."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for a in artifacts:
        if verbose:
            print(f"exporting {Path(a).name} ...", flush=True)
        rows.append(export_slide(a, out_dir, cfg, quality, min_downsample, verbose))

    # Order the picker cleanest-first, so clicking down the list walks from
    # a negative control to a severe animal. That ordering is the argument.
    rows.sort(key=lambda r: r["fat_percent"])
    (out_dir / "slides.json").write_text(json.dumps(
        [{"name": r["name"], "dir": r["dir"], "fat_percent": r["fat_percent"]}
         for r in rows], indent=2))

    shutil.copy(STATIC / "openseadragon.min.js", out_dir / "openseadragon.min.js")
    shutil.copytree(STATIC / "images", out_dir / "images", dirs_exist_ok=True)
    shutil.copy(STATIC / "LICENSE.openseadragon.txt",
                out_dir / "LICENSE.openseadragon.txt")
    shutil.copy(STATIC / "static_viewer.html", out_dir / "index.html")

    total = sum(r["bytes"] for r in rows)
    if verbose:
        print(f"\nsite: {len(rows)} slides, {sum(r['tiles'] for r in rows)} tiles, "
              f"{total / 1e6:.1f} MB total", flush=True)
    return {"slides": rows, "bytes": total}


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("artifacts", nargs="+",
                   help="precomputed slide directories (each with meta.json)")
    p.add_argument("--out", required=True)
    p.add_argument("--quality", type=int, default=72)
    p.add_argument("--min-downsample", type=int, default=2,
                   help="deepest exported level, as a level-0 downsample. "
                        "2 is half resolution (~1 um/px); 1 is full.")
    a = p.parse_args(argv)
    r = export_site(a.artifacts, a.out, quality=a.quality,
                    min_downsample=a.min_downsample)
    print(json.dumps({"total_mb": round(r["bytes"] / 1e6, 1),
                      "slides": [s["name"] for s in r["slides"]]}, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
