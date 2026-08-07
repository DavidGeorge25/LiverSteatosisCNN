"""Configuration for the steatosis pseudo-labeling pipeline.

Every threshold and filter parameter lives here (or in a YAML file that
overrides these defaults). Nothing is hardcoded in the algorithm modules --
they all take a config object. CLI flags override YAML, YAML overrides these
defaults.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TissueConfig:
    """Tissue vs. background detection, run at a low pyramid level."""

    # Pyramid level used for detection. Level 2 on these Aperio slides is a
    # 16x downsample -> ~2241 x 2505, which fits in RAM comfortably.
    level: int = 2

    # "otsu" picks the saturation cutoff automatically; "fixed" uses
    # `saturation_threshold` directly (0-255). Otsu is the default because
    # stain intensity varies slide to slide.
    method: str = "otsu"
    saturation_threshold: int = 25

    # Pixels brighter than this in HSV V are forced to background regardless of
    # saturation. Guards against pale but slightly-colored glass/scan padding.
    max_value: int = 245

    # Morphological cleanup, in microns so the numbers mean something physical
    # and stay correct if the detection level changes.
    close_radius_um: float = 20.0
    open_radius_um: float = 10.0

    # Speck removal. The slides have visible dust/debris outside the tissue.
    # Anything smaller than this is dropped.
    min_object_area_um2: float = 50_000.0

    # Holes inside tissue smaller than this get filled. Kept modest so real
    # structures (large vessel lumens) are not swallowed -- and so fat droplets,
    # which are orders of magnitude smaller, are always filled in.
    max_hole_area_um2: float = 200_000.0

    # Keep only the N largest tissue components (0 = keep all that survive the
    # area filter). Useful when a slide has one main section plus junk.
    keep_largest_n: int = 0


@dataclass
class TilingConfig:
    """Level-0 tiling over the detected tissue."""

    tile_size: int = 512
    stride: int = 512  # == tile_size means non-overlapping
    level: int = 0
    min_tissue_fraction: float = 0.5
    save_tiles: bool = True


@dataclass
class FatConfig:
    """Fat droplet pseudo-labeling. (Phase 3 -- not yet wired up.)"""

    method: str = "fixed"  # "fixed" | "otsu"
    white_threshold: int = 200

    fill_holes: bool = True
    open_radius_px: int = 2
    close_radius_px: int = 2

    # Macrovesicular droplets ~5-50 um diameter -> area in um^2.
    min_area_um2: float = 20.0
    max_area_um2: float = 2000.0

    circularity_min: float = 0.65
    solidity_min: float = 0.85

    # Best-fit-ellipse eccentricity ceiling (0 = circle, 1 = line). Rejects
    # elongated vessel/sinusoidal lumens, which circularity does not catch:
    # circularity responds to boundary raggedness, not to elongation, so a
    # smooth 2.5:1 lumen scores ~0.87 and sails through. Macro class only --
    # the micro band is too small for a stable ellipse fit. 1.0 disables it.
    max_eccentricity: float = 0.80

    # Microvesicular steatosis: fine speckling below the macro min-area cutoff.
    include_microvesicular: bool = False
    micro_min_area_um2: float = 3.0
    micro_max_area_um2: float = 20.0
    micro_circularity_min: float = 0.5
    micro_solidity_min: float = 0.8

    # Drop components touching the tile border -- usually tears or the
    # tissue-edge gap rather than whole droplets.
    exclude_border: bool = False


@dataclass
class QCConfig:
    """Visual quality control outputs."""

    contact_sheet_tiles: int = 16
    contact_sheet_cols: int = 4
    random_seed: int = 0
    outline_color: tuple[int, int, int] = (255, 0, 0)  # tissue mask overlay
    macro_color: tuple[int, int, int] = (255, 0, 0)  # macrovesicular droplets
    micro_color: tuple[int, int, int] = (0, 190, 255)  # microvesicular droplets
    outline_thickness: int = 2
    thumbnail_max_dim: int = 2000
    save_per_tile_qc: bool = True
    histogram_bins: int = 12


@dataclass
class Config:
    slide_path: str = ""
    output_dir: str = "outputs"
    limit: int = 0  # 0 = no limit; otherwise process only the first N tiles

    tissue: TissueConfig = field(default_factory=TissueConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    fat: FatConfig = field(default_factory=FatConfig)
    qc: QCConfig = field(default_factory=QCConfig)

    # ---- serialization -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        _apply_dict(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Dump the fully-resolved config next to the results, so any output
        directory records exactly which parameters produced it."""
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply dotted-key overrides, e.g. {"tissue.level": 1}.

        Values that are None are ignored, which lets argparse defaults of None
        mean "not specified on the command line".
        """
        for dotted, value in overrides.items():
            if value is None:
                continue
            target: Any = self
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = getattr(target, part)
            leaf = parts[-1]
            if not hasattr(target, leaf):
                raise KeyError(f"unknown config key: {dotted}")
            setattr(target, leaf, value)


def _apply_dict(obj: Any, data: dict[str, Any]) -> None:
    valid = {f.name for f in fields(obj)}
    for key, value in data.items():
        if key not in valid:
            raise KeyError(
                f"unknown config key '{key}' for {type(obj).__name__}; "
                f"valid keys: {sorted(valid)}"
            )
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(obj, key, value)
