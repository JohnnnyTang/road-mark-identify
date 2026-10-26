"""Per-cell channel statistics for one block (§4.3).

Channels (one grid, one CRS, one pixel size):
  int_max, int_mean, int_p90  corrected intensity of ground points        float32, NaN = no data
  density                     ground point count                          uint16, 0 = no points
  coverage                    1 where density >= coverage_min_points      uint8
  hag_max                     max height of low non-ground points          float32, 0 on bare
                              (0 < HAG <= low_nonground_max_hag)           ground, NaN = no data
  ground_z                    mean ground height (for curb steps)          float32, NaN = no data

A cell without points is no-data, never zero intensity.
"""

from __future__ import annotations

import numpy as np

CHANNELS: dict[str, tuple[str, float | None]] = {
    "int_max": ("float32", float("nan")),
    "int_mean": ("float32", float("nan")),
    "int_p90": ("float32", float("nan")),
    "density": ("uint16", None),
    "coverage": ("uint8", None),
    "hag_max": ("float32", float("nan")),
    "ground_z": ("float32", float("nan")),
}


def _group(flat: np.ndarray, values: np.ndarray):
    """Sort values by cell then value; return sorted values, group starts, counts, cell ids."""
    order = np.lexsort((values, flat))
    f, v = flat[order], values[order]
    cells, starts, counts = np.unique(f, return_index=True, return_counts=True)
    return v, starts, counts, cells


def cell_stats(row: np.ndarray, col: np.ndarray, shape: tuple[int, int], intensity: np.ndarray,
               z: np.ndarray, quantile: float = 0.9) -> dict[str, np.ndarray]:
    """Intensity/density/ground_z channels from the ground points inside a window."""
    h, w = shape
    out = {k: np.full(shape, np.nan, dtype=np.float32)
           for k in ("int_max", "int_mean", "int_p90", "ground_z")}
    out["density"] = np.zeros(shape, dtype=np.uint16)
    if len(row) == 0:
        return out
    flat = row * w + col
    v, starts, counts, cells = _group(flat, intensity.astype(np.float64))
    r, c = np.divmod(cells, w)
    out["int_max"][r, c] = v[starts + counts - 1]
    out["int_mean"][r, c] = np.add.reduceat(v, starts) / counts
    k = starts + np.ceil(quantile * counts).astype(np.int64) - 1  # nearest-rank percentile
    out["int_p90"][r, c] = v[np.clip(k, starts, starts + counts - 1)]
    out["density"][r, c] = np.minimum(counts, np.iinfo(np.uint16).max)
    zsum = np.bincount(flat, weights=z, minlength=h * w).reshape(shape)
    has = out["density"] > 0
    out["ground_z"][has] = (zsum[has] / out["density"][has]).astype(np.float32)
    return out


def hag_max(row: np.ndarray, col: np.ndarray, shape: tuple[int, int], hag: np.ndarray,
            ground_density: np.ndarray) -> np.ndarray:
    """Max HAG of low non-ground points; 0 on observed bare ground; NaN where nothing observed."""
    out = np.full(shape, np.nan, dtype=np.float32)
    out[ground_density > 0] = 0.0
    if len(row):
        acc = np.full(shape[0] * shape[1], -np.inf)
        np.maximum.at(acc, row * shape[1] + col, hag)
        acc = acc.reshape(shape)
        seen = np.isfinite(acc)
        out[seen] = np.maximum(np.nan_to_num(out[seen], nan=0.0), acc[seen])
    return out
