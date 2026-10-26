"""Stage 1 acceptance checks (§12 step 1).

* cross-pass intensity consistency on the same markings, within ``consistency_tol``;
* the density channel separates occlusions (no-data) from observed ground.
"""

from __future__ import annotations

import numpy as np


def pair_rel_diffs(mean_by_pass: dict[int, np.ndarray], paint: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Relative difference of corrected intensity between every pair of passes on paint cells.

    ``mean_by_pass[p]`` is an int_mean raster built from pass ``p`` only (NaN = not seen).
    """
    pids = sorted(mean_by_pass)
    out = {}
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            both = paint & np.isfinite(mean_by_pass[a]) & np.isfinite(mean_by_pass[b])
            va, vb = mean_by_pass[a][both], mean_by_pass[b][both]
            out[f"{a}-{b}"] = np.abs(va - vb) / np.maximum((va + vb) / 2, 1e-6)
    return out


def summarize_pairs(diffs: dict[str, list[np.ndarray]], tol: float, min_cells: int = 50) -> dict:
    """Pass/fail of cross-pass consistency; the worst pair decides."""
    pairs = {}
    for k, parts in diffs.items():
        rel = np.concatenate(parts) if parts else np.empty(0)
        if len(rel) < min_cells:
            continue
        pairs[k] = {"cells": int(len(rel)), "median_rel_diff": float(np.median(rel)),
                    "p90_rel_diff": float(np.quantile(rel, 0.9))}
    worst = max((p["median_rel_diff"] for p in pairs.values()), default=float("nan"))
    return {"pairs": pairs, "worst_median_rel_diff": worst, "tolerance": tol,
            "passed": bool(pairs) and worst <= tol}


def coverage_report(density: np.ndarray, coverage: np.ndarray, road: np.ndarray | None = None
                    ) -> dict:
    out = {"cells": int(density.size),
           "observed_fraction": float((coverage > 0).mean()),
           "density_p50": float(np.median(density[density > 0])) if (density > 0).any() else 0.0}
    if road is not None and road.any():
        out["road_nodata_fraction"] = float((coverage[road] == 0).mean())
    return out
