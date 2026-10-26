"""Ground filtering (§4.1) and height above ground.

Methods:
  existing  - trust the delivered classification (ASPRS class 2 by default). Verify first!
  csf       - Cloth Simulation Filter (``uv sync --extra ground``). Preferred if no trusted class.
  grid_min  - morphological lowest-point filter. Debug/fallback only: it has no notion of
              occlusion and accepts low objects on flat roads.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from roadmark.config import GroundCfg


def classify_ground(pts: dict[str, np.ndarray], cfg: GroundCfg) -> np.ndarray:
    """Boolean ground mask for the points of one block (block + halo)."""
    if cfg.method == "existing":
        return np.isin(pts["classification"], cfg.ground_classes)
    if cfg.method == "csf":
        return _csf(pts, cfg)
    if cfg.method == "grid_min":
        return _grid_min(pts, cfg)
    raise ValueError(cfg.method)


def _csf(pts: dict[str, np.ndarray], cfg: GroundCfg) -> np.ndarray:
    try:
        import CSF
    except ImportError as e:
        raise ImportError("CSF ground filter needs: uv sync --extra ground") from e
    p = cfg.csf
    csf = CSF.CSF()
    csf.params.bSloopSmooth = p.slope_smooth
    csf.params.cloth_resolution = p.cloth_resolution
    csf.params.rigidness = p.rigidness
    csf.params.class_threshold = p.class_threshold
    csf.params.time_step = p.time_step
    csf.params.interations = p.iterations  # sic: the binding's spelling
    # CSF works in float; shift to a local origin to keep precision.
    xyz = np.column_stack([pts["x"] - pts["x"].min(), pts["y"] - pts["y"].min(), pts["z"]])
    csf.setPointCloud(xyz)
    ground, non_ground = CSF.VecInt(), CSF.VecInt()
    try:
        csf.do_filtering(ground, non_ground, exportCloth=False)
    except TypeError:
        csf.do_filtering(ground, non_ground)
    mask = np.zeros(len(xyz), dtype=bool)
    mask[np.asarray(ground, dtype=np.int64)] = True
    return mask


def _grid_min(pts: dict[str, np.ndarray], cfg: GroundCfg) -> np.ndarray:
    p = cfg.grid_min
    x, y, z = pts["x"], pts["y"], pts["z"]
    cx = np.floor((x - x.min()) / p.cell).astype(np.int64)
    cy = np.floor((y - y.min()) / p.cell).astype(np.int64)
    shape = (cy.max() + 1, cx.max() + 1)
    zmin = np.full(shape, np.inf)
    np.minimum.at(zmin, (cy, cx), z)
    ref = ndimage.minimum_filter(zmin, size=p.window_cells, mode="nearest")
    ref[~np.isfinite(ref)] = np.inf
    tol = p.max_height + p.slope_tol * p.window_cells * p.cell / 2
    return (z - ref[cy, cx]) <= tol


def ground_dem(x: np.ndarray, y: np.ndarray, z: np.ndarray, cell: float,
               origin: tuple[float, float], shape: tuple[int, int]) -> np.ndarray:
    """Mean ground height per DEM cell, holes filled from the nearest valid cell.

    ``origin`` = (x_west, y_north), ``shape`` = (rows, cols). Returns NaN only if empty.
    """
    col = np.clip(np.floor((x - origin[0]) / cell).astype(np.int64), 0, shape[1] - 1)
    row = np.clip(np.floor((origin[1] - y) / cell).astype(np.int64), 0, shape[0] - 1)
    s = np.zeros(shape)
    n = np.zeros(shape)
    np.add.at(s, (row, col), z)
    np.add.at(n, (row, col), 1)
    dem = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    holes = ~np.isfinite(dem)
    if holes.all():
        return dem
    if holes.any():
        idx = ndimage.distance_transform_edt(holes, return_distances=False, return_indices=True)
        dem = dem[tuple(idx)]
    return dem


def height_above_ground(pts: dict[str, np.ndarray], ground: np.ndarray, cell: float) -> np.ndarray:
    """HAG for every point from a ground DEM built on this block's ground points."""
    x, y = pts["x"], pts["y"]
    origin = (np.floor(x.min() / cell) * cell, np.ceil(y.max() / cell) * cell)
    shape = (int(np.ceil((origin[1] - y.min()) / cell)) + 1,
             int(np.ceil((x.max() - origin[0]) / cell)) + 1)
    if not ground.any():
        return np.full(len(x), np.nan)
    dem = ground_dem(x[ground], y[ground], pts["z"][ground], cell, origin, shape)
    col = np.clip(np.floor((x - origin[0]) / cell).astype(np.int64), 0, shape[1] - 1)
    row = np.clip(np.floor((origin[1] - y) / cell).astype(np.int64), 0, shape[0] - 1)
    return pts["z"] - dem[row, col]
