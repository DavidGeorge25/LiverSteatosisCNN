"""Shared configuration machinery and the core (feature-agnostic) sections.

Every threshold and filter parameter lives here or in a feature's own config
module -- or in a YAML file that overrides those defaults. Nothing is hardcoded
in the algorithm modules; they all take a config object. CLI flags override
YAML, YAML overrides these defaults.

Config files layer: `--config configs/core.yaml --config configs/steatosis.yaml`
applies them left to right, so shared settings live in one file and each
feature's thresholds in another. A single combined file still works unchanged.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SlideConfig:
    """How to open a slide, and what to do when it carries no physical scale.

    Aperio SVS embeds microns-per-pixel, so these stay None for the MASH and
    CCl4 cohorts. Plain TIFF frequently does not, and every filter in this
    project is specified in microns -- so rather than assume a magnification,
    the reader refuses to open such a slide until `mpp_x`/`mpp_y` are set here.
    A wrong guess would silently rescale every area threshold.
    """

    # Microns per pixel at LEVEL 0. Set only for formats with no embedded
    # scale; when the slide has its own, that wins unless `override_mpp`.
    mpp_x: float | None = None
    mpp_y: float | None = None

    # Use the values above even when the file carries its own scale. Off by
    # default so a config left over from a TIFF run cannot quietly rescale an
    # SVS slide that knows its own MPP.
    override_mpp: bool = False

    # Guard for the tifffile fallback, which (without zarr installed) can only
    # crop out of a fully decoded level. Levels above this are refused rather
    # than silently exhausting RAM.
    max_in_memory_megapixels: float = 64.0


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
    """Level-0 tiling over the detected tissue. Shared by all three features."""

    tile_size: int = 512
    stride: int = 512  # == tile_size means non-overlapping
    level: int = 0
    min_tissue_fraction: float = 0.5
    save_tiles: bool = True


@dataclass
class QCConfig:
    """Visual quality control outputs."""

    contact_sheet_tiles: int = 16
    contact_sheet_cols: int = 4
    random_seed: int = 0
    outline_color: tuple[int, int, int] = (255, 0, 0)  # tissue mask overlay
    # Droplet class colors. Steatosis-specific, but they stay on the shared QC
    # section because every tuned config and every `config_used.yaml` already
    # written has them under `qc:` -- moving them would break those files.
    macro_color: tuple[int, int, int] = (255, 0, 0)  # macrovesicular droplets
    micro_color: tuple[int, int, int] = (0, 190, 255)  # microvesicular droplets
    outline_thickness: int = 2
    thumbnail_max_dim: int = 2000
    save_per_tile_qc: bool = True
    histogram_bins: int = 12


@dataclass
class StainConfig:
    """Stain normalization at preprocessing. OFF by default, on purpose.

    Staining batch is one-to-one with experiment in this collection, so
    normalization is not a free win -- it is an intervention whose effect on
    cross-batch generalisation has to be measured, not assumed. Leaving it off
    by default means the first numbers produced are the un-normalized baseline
    that any later claim about normalization has to beat. See
    `mashpath/train/stain.py`.
    """

    enabled: bool = False
    method: str = "none"           # none | macenko | reinhard
    reference_image: str = ""      # fit the target appearance from this tile;
                                   # empty = literature H&E reference vectors
    beta: float = 0.15             # optical-density floor: below this is glass
    alpha: float = 1.0             # angle percentile for the stain vectors


@dataclass
class AugmentConfig:
    """Colour augmentation at training time. ON by default, also on purpose.

    The mirror of StainConfig: normalization is the intervention that has to
    prove itself, augmentation is the one that has to be justified if it is
    turned OFF, because without it colour remains an available shortcut and
    colour is exactly what distinguishes the batches from each other. Ranges
    are full widths; see `mashpath/train/augment.py`.
    """

    strength: str = "strong"       # none | mild | strong
    hue: float = 0.10
    saturation: float = 0.60
    brightness: float = 0.50
    contrast: float = 0.50
    gamma: float = 0.40
    stain_sigma: float = 0.25
    stain_bias: float = 0.05
    probability: float = 0.9


@dataclass
class ReviewConfig:
    """Export of machine-generated candidates for pathologist confirmation.

    Ballooning and lobular inflammation have no threshold that finds them, so
    the pipeline proposes and a pathologist disposes. Context size is in
    microns: a ballooned hepatocyte is only callable against its neighbours, so
    the crop has to contain them at a fixed physical size regardless of level.
    """

    context_um: float = 150.0
    max_per_slide: int = 200
    strata: int = 4  # score bands sampled evenly, so the extremes appear
    seed: int = 0
    save_overlay: bool = True
    overlay_color: tuple[int, int, int] = (255, 0, 0)
    overlay_thickness: int = 2


class ConfigBase:
    """YAML load/save and dotted-key overrides for a root config dataclass."""

    # ---- serialization -------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path | list[str | Path]):
        """Load one YAML file, or layer several left to right."""
        paths = path if isinstance(path, (list, tuple)) else [path]
        cfg = cls()
        for p in paths:
            with open(p) as fh:
                data = yaml.safe_load(fh) or {}
            _apply_dict(cfg, data)
        return cfg

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        cfg = cls()
        _apply_dict(cfg, data)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        """Dump the fully-resolved config next to the results, so any output
        directory records exactly which parameters produced it."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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


@dataclass
class CoreConfig(ConfigBase):
    """The sections every feature shares. Feature configs extend this."""

    slide_path: str = ""
    output_dir: str = "outputs"
    limit: int = 0  # 0 = no limit; otherwise process only the first N tiles

    slide: SlideConfig = field(default_factory=SlideConfig)
    tissue: TissueConfig = field(default_factory=TissueConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    stain: StainConfig = field(default_factory=StainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
