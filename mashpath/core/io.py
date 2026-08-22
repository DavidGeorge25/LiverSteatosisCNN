"""Output paths and writing.

One place decides where anything lands, so three features writing to the same
slide directory cannot collide and a reader can find a result without knowing
which feature produced it:

    outputs/<slide_id>/
        tiles/                  tile images, shared by every feature
        steatosis/              per-tile CSV, masks/, review/
        ballooning/
        inflammation/
        qc/                     contact sheets and overlays, per feature
        config_used.yaml        the fully resolved config for this run
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

FEATURES = ("steatosis", "ballooning", "inflammation")


@dataclass(frozen=True)
class SlideOutputs:
    """Where every artifact for one slide goes."""

    root: Path
    slide_id: str

    @classmethod
    def for_slide(cls, output_dir: str | Path, slide_id: str) -> "SlideOutputs":
        return cls(root=Path(output_dir) / slide_id, slide_id=slide_id)

    # ---- directories ----------------------------------------------------

    @property
    def tiles_dir(self) -> Path:
        return self.root / "tiles"

    @property
    def qc_dir(self) -> Path:
        return self.root / "qc"

    def feature_dir(self, feature: str) -> Path:
        _check_feature(feature)
        return self.root / feature

    def masks_dir(self, feature: str) -> Path:
        return self.feature_dir(feature) / "masks"

    def review_dir(self, feature: str) -> Path:
        """Where candidates go to be confirmed by a pathologist."""
        return self.feature_dir(feature) / "review"

    def feature_qc_dir(self, feature: str) -> Path:
        _check_feature(feature)
        return self.qc_dir / feature

    def mkdirs(self, feature: str | None = None) -> "SlideOutputs":
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        if feature:
            self.feature_dir(feature).mkdir(parents=True, exist_ok=True)
        return self

    # ---- writing --------------------------------------------------------

    def write_table(
        self, rows: Iterable[dict[str, Any]] | Any, feature: str, name: str
    ) -> Path:
        """Write a CSV under the feature's directory. Accepts rows or a
        DataFrame."""
        import pandas as pd

        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        path = self.feature_dir(feature) / _with_suffix(name, ".csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

    def write_mask(self, mask: np.ndarray, feature: str, name: str) -> Path:
        """Write a binary mask as a black/white PNG."""
        from .viz import save_rgb

        m = (np.asarray(mask).astype(bool).astype(np.uint8) * 255)
        return save_rgb(
            self.masks_dir(feature) / _with_suffix(name, ".png"),
            np.dstack([m, m, m]),
        )

    def write_tile(self, rgb: np.ndarray, name: str) -> Path:
        from .viz import save_rgb

        return save_rgb(self.tiles_dir / _with_suffix(name, ".png"), rgb)

    def write_qc(self, rgb: np.ndarray, feature: str | None, name: str) -> Path:
        from .viz import save_rgb

        d = self.feature_qc_dir(feature) if feature else self.qc_dir
        return save_rgb(d / _with_suffix(name, ".png"), rgb)

    def write_config(self, cfg: Any) -> Path:
        path = self.root / "config_used.yaml"
        cfg.save(path)
        return path

    def log_path(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        return self.root / "run.log"


def _check_feature(feature: str) -> None:
    if feature not in FEATURES:
        raise ValueError(f"unknown feature {feature!r}; expected one of {FEATURES}")


def _with_suffix(name: str, suffix: str) -> str:
    return name if name.endswith(suffix) else name + suffix


def write_index(outputs: SlideOutputs, rows: Iterable[dict[str, Any]], name: str) -> Path:
    """A table belonging to the slide rather than to any one feature."""
    import pandas as pd

    outputs.root.mkdir(parents=True, exist_ok=True)
    path = outputs.root / _with_suffix(name, ".csv")
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path
