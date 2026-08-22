"""The candidate record: one thing a detector wants a pathologist to judge.

Deliberately feature-agnostic. A ballooning candidate and an inflammatory focus
carry different measurements, so those go in `measurements` rather than in
fixed columns -- the review package, the manifest, and the confirmed-set loader
then work for both, and for whatever feature comes fourth.

Coordinates are LEVEL-0 pixels throughout, which is the only frame that stays
meaningful across pyramid levels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Candidate:
    """One proposed object, with the numbers that made it a proposal."""

    slide_id: str
    feature: str
    x: int  # level-0 centroid
    y: int
    width: int  # level-0 bounding box
    height: int
    score: float = 0.0  # detector confidence; only ordering is meaningful
    tile_x: int = 0
    tile_y: int = 0
    measurements: dict[str, float] = field(default_factory=dict)

    @property
    def candidate_id(self) -> str:
        return f"{self.feature}_x{self.x:06d}_y{self.y:06d}"

    def to_row(self) -> dict[str, Any]:
        row = {
            "candidate_id": self.candidate_id,
            "slide": self.slide_id,
            "feature": self.feature,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "score": round(self.score, 6),
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
        }
        row.update({f"m_{k}": v for k, v in self.measurements.items()})
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Candidate":
        return cls(
            slide_id=str(row["slide"]),
            feature=str(row["feature"]),
            x=int(row["x"]),
            y=int(row["y"]),
            width=int(row["width"]),
            height=int(row["height"]),
            score=float(row.get("score", 0.0)),
            tile_x=int(row.get("tile_x", 0)),
            tile_y=int(row.get("tile_y", 0)),
            measurements={
                k[2:]: float(v) for k, v in row.items()
                if k.startswith("m_") and v not in (None, "")
            },
        )


@dataclass
class CandidateSet:
    """All candidates one detector produced for one slide."""

    slide_id: str
    feature: str
    candidates: list[Candidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.candidates)

    def __iter__(self) -> Iterator[Candidate]:
        return iter(self.candidates)

    def extend(self, items: Iterable[Candidate]) -> None:
        self.candidates.extend(items)

    def to_rows(self) -> list[dict[str, Any]]:
        return [c.to_row() for c in self.candidates]
