"""Redaction helpers: what the CLI prints must be safe to paste anywhere (DATA_POLICY.md).

Default output describes data by *size and shape* (extent dimensions, counts, densities),
never by *location* (absolute coordinates, bounds, origins). ``--show-coords`` opts in,
locally, when a person needs the real numbers.
"""

from __future__ import annotations

from collections.abc import Sequence


def extent_summary(bounds: Sequence[float], show_coords: bool = False) -> dict:
    """bounds = (minx, miny, maxx, maxy[, minz, maxz])."""
    minx, miny, maxx, maxy = bounds[:4]
    out = {"width_m": round(maxx - minx, 2), "height_m": round(maxy - miny, 2),
           "area_m2": round((maxx - minx) * (maxy - miny), 1)}
    if len(bounds) >= 6:
        out["z_range_m"] = round(bounds[5] - bounds[4], 2)
    if show_coords:
        out["bounds"] = [float(b) for b in bounds]
    return out
