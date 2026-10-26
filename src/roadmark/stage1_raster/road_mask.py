"""Road-surface prior from curb steps (§4.4).

The mask is a *prior*, not a filter: later stages flag paint outside it, never delete it
(islands and medians carry markings).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from roadmark.config import RoadMaskCfg


def curb_candidates(ground_z: np.ndarray, hag_max: np.ndarray, pixel_size: float,
                    cfg: RoadMaskCfg) -> np.ndarray:
    """Cells where the local height step matches a curb (curb_step_min..curb_step_max)."""
    size = max(3, int(round(cfg.step_window_m / pixel_size)) | 1)
    z = np.where(np.isfinite(ground_z), ground_z, np.nan)
    zmax = ndimage.maximum_filter(np.nan_to_num(z, nan=-np.inf), size=size)
    zmin = ndimage.minimum_filter(np.nan_to_num(z, nan=np.inf), size=size)
    step = zmax - zmin
    step[~np.isfinite(step)] = 0.0
    hag = np.nan_to_num(hag_max, nan=0.0)
    in_range = lambda s: (s >= cfg.curb_step_min) & (s <= cfg.curb_step_max)  # noqa: E731
    return in_range(step) | in_range(hag)


def road_surface(ground_z: np.ndarray, hag_max: np.ndarray, coverage: np.ndarray,
                 pixel_size: float, cfg: RoadMaskCfg) -> tuple[np.ndarray, np.ndarray]:
    """Return (road_mask, curb_mask). Road = large connected low regions bounded by curbs.

    Of the regions separated by curbs, the one(s) whose mean height is lowest relative to
    their neighbours are carriageway; raised regions (sidewalks) are excluded. Regions below
    ``min_road_area_m2`` are dropped.
    """
    curb = curb_candidates(ground_z, hag_max, pixel_size, cfg)
    curb = ndimage.binary_dilation(curb, iterations=1)
    # Close sampling pinholes in coverage *before* cutting curbs, so closing can't bridge them.
    observed = ndimage.binary_closing(coverage > 0, iterations=2)
    free = observed & ~curb
    labels, n = ndimage.label(free)
    if n == 0:
        return np.zeros_like(free), curb
    min_px = cfg.min_road_area_m2 / pixel_size**2
    areas = ndimage.sum_labels(free, labels, index=np.arange(1, n + 1))
    zmean = ndimage.mean(np.nan_to_num(ground_z, nan=0.0), labels, index=np.arange(1, n + 1))
    big = np.flatnonzero(areas >= min_px) + 1
    if len(big) == 0:
        return np.zeros_like(free), curb
    # Carriageway: big regions within one curb height of the lowest big region.
    zb = zmean[big - 1]
    keep = big[zb <= zb.min() + cfg.curb_step_min]
    return np.isin(labels, keep), curb
