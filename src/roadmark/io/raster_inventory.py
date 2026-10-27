"""Inventory of delivered raster tiles (Roadmap 1.8): one summary per channel folder.

Answers, without printing coordinates: pixel size and dtype, how no-data is encoded, whether
the tiles share one grid (also across channels), how tile names relate to tile positions, and
value distributions (sampled per tile, so thousands of tiles stay cheap).

Georeferencing may come from world files (``.tfw``) without a CRS; rasterio reads those.
"""

from __future__ import annotations

import glob
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from pydantic import BaseModel, Field

QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


class RasterInventoryCfg(BaseModel):
    groups: dict[str, list[str]]  # name -> globs relative to the data root
    sample_per_tile: int = 20_000
    seed: int = 0
    workers: int = Field(default=4, exclude=True)


def tile_name_index(name: str) -> tuple[int, int] | None:
    """'-2_5.tif' or 'grid_-2_5_g.las' -> (-2, 5); None if no two integers."""
    nums = [p for p in Path(name).name.split(".", 1)[0].split("_") if p.lstrip("-").isdigit()]
    return (int(nums[0]), int(nums[1])) if len(nums) >= 2 else None


def tile_stats(path: Path, n_sample: int, rng: np.random.Generator) -> dict:
    with rasterio.open(path) as ds:
        t = ds.transform
        arr = ds.read(masked=False)
        nodata = ds.nodata
        out = {
            "file": path.name,
            "shape": [ds.count, ds.height, ds.width],
            "dtype": ds.dtypes[0],
            "pixel_size": (round(t.a, 6), round(-t.e, 6)),
            "rotated": bool(t.b or t.d),
            "nodata": None if nodata is None else float(nodata),
            "crs": None if ds.crs is None else str(ds.crs.to_epsg() or "custom"),
            "colorinterp": [c.name for c in ds.colorinterp],
            "_origin": (t.c, t.f),  # kept for alignment checks only, dropped before reporting
        }
    flat = arr.reshape(arr.shape[0], -1)
    floating = np.issubdtype(flat.dtype, np.floating)
    nan = np.isnan(flat) if floating else np.zeros(flat.shape, bool)
    is_nd = (flat == nodata) if nodata is not None and not math.isnan(nodata) else nan
    valid = ~(is_nd | nan)
    out["px"] = int(flat.shape[1])
    out["nodata_px"] = [int(v) for v in (~valid).sum(1)]
    out["zero_px"] = [int(v) for v in (valid & (flat == 0)).sum(1)]
    out["_samples"] = []
    for b in range(flat.shape[0]):
        v = flat[b][valid[b]]
        if len(v) > n_sample:  # with replacement: a full shuffle of 10^7 pixels is too slow
            v = v[rng.integers(0, len(v), n_sample)]
        out["_samples"].append(v.astype(np.float64))
    return out


def _alignment(rows: list[dict]) -> dict:
    """Tile size, shared grid, and the name -> position relation. Counts and sizes only."""
    ps = Counter(r["pixel_size"] for r in rows).most_common(1)[0][0]
    w = Counter(r["shape"][2] * ps[0] for r in rows).most_common(1)[0][0]
    h = Counter(r["shape"][1] * ps[1] for r in rows).most_common(1)[0][0]
    on_grid = sum(abs(r["_origin"][0] / ps[0] - round(r["_origin"][0] / ps[0])) < 1e-3
                  and abs(r["_origin"][1] / ps[1] - round(r["_origin"][1] / ps[1])) < 1e-3
                  for r in rows)
    rel: Counter = Counter()
    for r in rows:
        idx = tile_name_index(r["file"])
        if idx is None:
            continue
        x0, ytop = r["_origin"]
        cands = {"(floor(minx/W), floor(miny/H))": (x0 / w, (ytop - h) / h),
                 "(floor(minx/W), floor(maxy/H))": (x0 / w, ytop / h),
                 "(floor(miny/H), floor(minx/W))": ((ytop - h) / h, x0 / w)}
        for k, (a, b) in cands.items():
            if (math.floor(a + 1e-6), math.floor(b + 1e-6)) == idx:
                rel[k] += 1
    return {"tile_size_m": [round(w, 3), round(h, 3)],
            "tiles_with_common_size": sum(r["shape"][2] * ps[0] == w and r["shape"][1] * ps[1] == h
                                          for r in rows),
            "origins_on_pixel_grid": on_grid,
            "name_matches_position": dict(rel)}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"files": 0}
    nb = max(r["shape"][0] for r in rows)
    px = sum(r["px"] for r in rows)
    bands = []
    for b in range(nb):
        rs = [r for r in rows if r["shape"][0] > b]
        s = np.concatenate([r["_samples"][b] for r in rs])
        bands.append({
            "nodata_frac": round(sum(r["nodata_px"][b] for r in rs) / px, 4),
            "zero_frac_of_valid": round(sum(r["zero_px"][b] for r in rs)
                                        / max(1, px - sum(r["nodata_px"][b] for r in rs)), 4),
            "distinct_values_in_sample": int(len(np.unique(s))),
            "min": float(s.min()) if len(s) else None,
            "max": float(s.max()) if len(s) else None,
            **({f"p{q * 100:g}": round(float(np.quantile(s, q)), 4) for q in QUANTILES}
               if len(s) else {}),
        })
    return {
        "files": len(rows),
        "shapes_bands_rows_cols": dict(Counter(str(r["shape"]) for r in rows).most_common(3)),
        "dtype": dict(Counter(r["dtype"] for r in rows)),
        "pixel_size_m": dict(Counter(str(r["pixel_size"]) for r in rows)),
        "rotated": sum(r["rotated"] for r in rows),
        "nodata_value": dict(Counter(str(r["nodata"]) for r in rows)),
        "crs": dict(Counter(str(r["crs"]) for r in rows)),
        "colorinterp": dict(Counter(",".join(r["colorinterp"]) for r in rows)),
        "grid": _alignment(rows),
        "bands": bands,
    }


def cross_group(groups: dict[str, list[dict]]) -> dict:
    """Do same-named tiles have identical grids (origin, pixel size, shape) across groups?"""
    keyed = {g: {r["file"]: (r["_origin"], r["pixel_size"], r["shape"][1:]) for r in rows}
             for g, rows in groups.items()}
    names = list(keyed)
    out = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = keyed[a].keys() & keyed[b].keys()
            out[f"{a}|{b}"] = {"common_tiles": len(common),
                               "identical_grid": sum(keyed[a][n] == keyed[b][n] for n in common)}
    return out


def _tile_job(args: tuple[Path, int, int]) -> dict:
    path, n_sample, seed = args
    return tile_stats(path, n_sample, np.random.default_rng(seed))


def run_raster_inventory(cfg: RasterInventoryCfg, root: Path) -> tuple[dict, list[Path]]:
    rows_by: dict[str, list[dict]] = {}
    files_all: list[Path] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for name, pats in cfg.groups.items():
            files = sorted({Path(f) for p in pats for f in glob.glob(str(
                Path(p) if Path(p).is_absolute() else root / p))})
            jobs = [(f, cfg.sample_per_tile, cfg.seed + i) for i, f in enumerate(files)]
            rows_by[name] = list(ex.map(_tile_job, jobs))
            files_all += files
    report = {"groups": {g: summarize(r) for g, r in rows_by.items()},
              "cross_group": cross_group(rows_by),
              "files": {g: [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
                        for g, rows in rows_by.items()}}
    return report, files_all
