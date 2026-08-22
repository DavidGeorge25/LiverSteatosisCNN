"""Inflammation feature CLI -- the stages that exist so far.

    python -m mashpath.features.inflammation.cli nuclei \
        --slide R26-122-1_HE_70.svs --config configs/inflammation.yaml --sample 60

Separate from `mashpath.cli` on purpose: that one exposes `inflammation` as a
candidate feature, which needs portal exclusion and focus clustering. This one
drives the parts that are built and reviewable now -- segmentation and the
class split -- so the cutoffs can be tuned before anything is built on them.

Every flag maps onto a config key; flags beat YAML, YAML beats the defaults.
"""

from __future__ import annotations

import argparse
import sys

from ...config import MashConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mashpath-inflammation",
        description="Lobular inflammation -- nucleus segmentation and class split",
    )
    sub = p.add_subparsers(dest="command", required=True)

    n = sub.add_parser(
        "nuclei",
        help="segment nuclei, split hepatocyte/immune, write QC",
        description="Stage 1-2. Writes a class overlay, the area-vs-intensity "
                    "distribution with the cutoffs drawn on it, and a map of "
                    "immune-classified nuclei over the slide thumbnail.",
    )
    n.add_argument("--slide", required=True, help="path to a whole slide image")
    n.add_argument("--config", action="append", default=None, metavar="YAML",
                   help="YAML config; repeat to layer files left to right")
    n.add_argument("--output-dir", default=None)

    g = n.add_argument_group("tile selection")
    g.add_argument("--limit", type=int, default=None,
                   help="process only the first N tiles in raster order (0 = all)")
    g.add_argument("--sample", type=int, default=0,
                   help="instead of raster order, draw N tiles uniformly from "
                        "the whole tissue. Use this for tuning -- the first N "
                        "tiles are all one corner of one lobe.")
    g.add_argument("--seed", type=int, default=0, help="sampling seed")
    g.add_argument("--tile-size", type=int, default=None)
    g.add_argument("--stride", type=int, default=None)
    g.add_argument("--min-tissue-fraction", type=float, default=None)

    g = n.add_argument_group("segmentation")
    g.add_argument("--model-name", default=None, help="StarDist pretrained model")
    g.add_argument("--model-dir", default=None, help="local model dir (offline)")
    g.add_argument("--prob-threshold", type=float, default=None,
                   help="StarDist objectness cutoff; lower finds more nuclei")
    g.add_argument("--nms-threshold", type=float, default=None)
    g.add_argument("--margin-px", type=int, default=None,
                   help="context read around each tile so boundary nuclei are "
                        "measured whole (0 disables)")
    g.add_argument("--min-nucleus-area-um2", type=float, default=None)
    g.add_argument("--max-nucleus-area-um2", type=float, default=None)

    g = n.add_argument_group("classification")
    g.add_argument("--intensity-channel", choices=["gray", "hematoxylin"],
                   default=None)
    g.add_argument("--immune-max-area-um2", type=float, default=None)
    g.add_argument("--immune-min-area-um2", type=float, default=None)
    g.add_argument("--hepatocyte-min-area-um2", type=float, default=None)
    g.add_argument("--immune-max-intensity", type=float, default=None,
                   help="grayscale cutoff; darker than this qualifies as immune")
    g.add_argument("--immune-min-hematoxylin", type=float, default=None,
                   help="haematoxylin OD cutoff; above this qualifies as immune")
    g.add_argument("--no-require-dark", dest="require_dark", action="store_false",
                   default=None, help="split on area alone, ignoring intensity")

    g = n.add_argument_group("QC")
    g.add_argument("--sample-tiles", type=int, default=None,
                   help="tiles drawn into the class-overlay contact sheet")
    g.add_argument("--fill-alpha", type=float, default=None,
                   help="0 = outline only")
    g.add_argument("--no-nucleus-csv", dest="save_nucleus_csv",
                   action="store_false", default=None,
                   help="skip the full per-nucleus table")
    return p


_OVERRIDES = {
    "slide_path": "slide",
    "output_dir": "output_dir",
    "limit": "limit",
    "tiling.tile_size": "tile_size",
    "tiling.stride": "stride",
    "tiling.min_tissue_fraction": "min_tissue_fraction",
    "inflammation.nuclei.model_name": "model_name",
    "inflammation.nuclei.model_dir": "model_dir",
    "inflammation.nuclei.prob_threshold": "prob_threshold",
    "inflammation.nuclei.nms_threshold": "nms_threshold",
    "inflammation.nuclei.margin_px": "margin_px",
    "inflammation.nuclei.min_area_um2": "min_nucleus_area_um2",
    "inflammation.nuclei.max_area_um2": "max_nucleus_area_um2",
    "inflammation.classify.intensity_channel": "intensity_channel",
    "inflammation.classify.immune_max_area_um2": "immune_max_area_um2",
    "inflammation.classify.immune_min_area_um2": "immune_min_area_um2",
    "inflammation.classify.hepatocyte_min_area_um2": "hepatocyte_min_area_um2",
    "inflammation.classify.immune_max_intensity": "immune_max_intensity",
    "inflammation.classify.immune_min_hematoxylin": "immune_min_hematoxylin",
    "inflammation.classify.require_dark": "require_dark",
    "inflammation.nucleus_qc.sample_tiles": "sample_tiles",
    "inflammation.nucleus_qc.fill_alpha": "fill_alpha",
    "inflammation.nucleus_qc.save_nucleus_csv": "save_nucleus_csv",
}


def load_config(args: argparse.Namespace) -> MashConfig:
    cfg = MashConfig.from_yaml(args.config) if args.config else MashConfig()
    cfg.apply_overrides(
        {key: getattr(args, attr, None) for key, attr in _OVERRIDES.items()}
    )
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args)

    if args.command == "nuclei":
        from .pipeline import run_nuclei

        run_nuclei(cfg, sample=args.sample, seed=args.seed)
        return 0
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
