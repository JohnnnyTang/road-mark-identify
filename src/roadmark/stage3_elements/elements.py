"""Paint element record shared by Stages 3-5."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shapely.geometry import Polygon

ElementType = Literal["stroke", "bar", "glyph", "hatch_stripe", "noise", "unknown"]
Evidence = Literal["observed", "inferred"]


@dataclass
class PaintElement:
    element_id: str
    geometry: Polygon
    length_m: float
    width_m: float
    orientation_deg: float  # long axis, 0 = east, counter-clockwise
    rel_orientation_deg: float | None  # vs local road direction; None in low-coherence zones
    area_m2: float
    solidity: float
    rectangularity: float
    skeleton_length_m: float
    skeleton_branches: int
    prob_mean: float
    prob_max: float
    nodata_buffer_fraction: float
    in_road_mask: bool
    etype: ElementType = "unknown"
    evidence: Evidence = "observed"
    merged_from: list[str] = field(default_factory=list)
    object_id: str | None = None  # set by Stage 4; None -> residual_paint
