"""Track model: centreline + linear-referenced intervals (§2.2, §8.2). OpenDRIVE-compatible."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from shapely.geometry import LineString

Pattern = Literal["solid", "dashed", "unknown"]
Role = Literal["lane_line", "edge_line", "centre_line", "guide", "area_boundary", "unknown"]


@dataclass
class TrackInterval:
    subline: int  # 0 single; 1 left, 2 right (relative to digitizing direction)
    s_start: float
    s_end: float
    lateral_offset_m: float
    pattern: Pattern
    width_m: float
    evidence: Literal["observed", "inferred"]
    confidence: float
    dash_len_m: float | None = None
    gap_len_m: float | None = None
    colour: Literal["white", "yellow", "unknown"] = "unknown"


@dataclass
class Track:
    track_id: str
    centreline: LineString
    role: Role = "unknown"
    intervals: list[TrackInterval] = field(default_factory=list)
    element_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    review_status: str = "needs_review"
    provenance: str = ""

    @property
    def n_sublines_max(self) -> int:
        return 2 if any(i.subline > 0 for i in self.intervals) else 1
