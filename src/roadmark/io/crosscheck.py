"""Cross-check delivered raster tiles against each other and the point tiles (Roadmap 1.8).

For a sample of raster tiles, with all channels read together:
  - value distributions conditioned on density > 0 vs density == 0, which shows whether 0
    encodes no-data (three-state evidence needs to know)
  - grid conventions: origin offset in pixels (world files sometimes give corners, not centres)
    and which rule maps tile names to positions
  - registration with the ground points: share of points that land on density > 0 pixels
    (≈ 1 if the frames agree, ≈ the occupied fraction if they don't), and how raster values
    relate to point values (rank correlation of intensity, linear slope of height vs z)

Redacted: shares, correlations, slopes and histograms only. No positions or elevations.
"""

from __future__ import annotations

import glob
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import rasterio
from pydantic import BaseModel, Field
from scipy import ndimage, stats

from roadmark.io.raster_inventory import tile_name_index


class CrosscheckCfg(BaseModel):
    channels: dict[str, str]  # channel -> glob of its tiles (relative to the data root)
    density_channel: str = "density"
    points: list[str] = Field(default_factory=list)  # ground point globs
    n_tiles: int = 12  # sampled: the largest density tiles (most content)
    max_points_per_tile: int = 2_000_000
    seed: int = 0
    workers: int = Field(default=3, exclude=True)


def _glob(root: Path, pat: str) -> list[Path]:
    return sorted(Path(f) for f in glob.glob(str(Path(pat) if Path(pat).is_absolute()
                                                 else root / pat)))


def _name_rules(idx: tuple[int, int], x0: float, ytop: float, w: float, h: float, ps: float
                ) -> list[str]:
    """Which candidate rules reproduce the tile name index from the tile position."""
    hits = []
    for shift_name, s in (("", 0.0), ("+half_px", ps / 2), ("-half_px", -ps / 2)):
        for ref_name, yref in (("miny", ytop - h), ("maxy", ytop)):
            for op_name, op in (("floor", math.floor), ("round", round)):
                ix, iy = op((x0 + s) / w + 1e-9), op((yref + s) / h + 1e-9)
                if (ix, iy) == idx:
                    hits.append(f"{op_name}(minx/W),{op_name}({ref_name}/H){shift_name}")
                if (iy, ix) == idx:
                    hits.append(f"swapped:{op_name}({ref_name}/H),{op_name}(minx/W){shift_name}")
    return hits


def tile_check(args: tuple[str, dict[str, Path], str, list[tuple[Path, tuple]], int, int]) -> dict:
    name, paths, dens_ch, point_tiles, max_pts, seed = args
    rng = np.random.default_rng(seed)
    arrs = {}
    for ch, p in paths.items():
        with rasterio.open(p) as ds:
            arrs[ch] = ds.read(1)
            t, shape = ds.transform, (ds.height, ds.width)
    ps = t.a
    x0, ytop = t.c, t.f
    w, h = shape[1] * ps, shape[0] * ps
    dens = arrs[dens_ch]
    occ = dens > 0
    out: dict = {"name": name, "px": int(dens.size), "occupied_px": int(occ.sum()),
                 "origin_offset_px": [round((x0 / ps) % 1, 3), round((ytop / ps) % 1, 3)],
                 "name_rules": [], "hist_occ": {}, "hist_empty": {}}
    idx = tile_name_index(name)
    if idx is not None:
        out["name_rules"] = _name_rules(idx, x0, ytop, w, h, ps)
    for ch, a in arrs.items():
        out["hist_occ"][ch] = np.bincount(a[occ].ravel(), minlength=256)[:256]
        out["hist_empty"][ch] = np.bincount(a[~occ].ravel(), minlength=256)[:256]

    # Registration against ground points overlapping this tile.
    xs, ys, zs, its = [], [], [], []
    for p, (bx0, by0, bx1, by1) in point_tiles:
        if bx1 < x0 or bx0 > x0 + w or by1 < ytop - h or by0 > ytop:
            continue
        las = laspy.read(p)
        x, y = np.asarray(las.x), np.asarray(las.y)
        m = (x >= x0) & (x < x0 + w) & (y > ytop - h) & (y <= ytop)
        xs.append(x[m])
        ys.append(y[m])
        zs.append(np.asarray(las.z)[m])
        its.append(np.asarray(las.intensity, np.float64)[m])
    if xs and sum(len(v) for v in xs):
        x, y, z, it = (np.concatenate(v) for v in (xs, ys, zs, its))
        if len(x) > max_pts:
            sel = rng.choice(len(x), max_pts, replace=False)
            x, y, z, it = x[sel], y[sel], z[sel], it[sel]
        col = np.clip(np.floor((x - x0) / ps).astype(np.int64), 0, shape[1] - 1)
        row = np.clip(np.floor((ytop - y) / ps).astype(np.int64), 0, shape[0] - 1)
        occ3 = ndimage.binary_dilation(occ, iterations=1)
        on = occ[row, col]
        reg = {"points": int(len(x)), "on_occupied": round(float(on.mean()), 4),
               "on_occupied_within_1px": round(float(occ3[row, col].mean()), 4),
               "on_occupied_if_shifted_px": {}}
        # If the world file holds pixel corners, a half-pixel shift raises the hit rate.
        for dx in (-0.5, 0.0, 0.5):
            for dy in (-0.5, 0.0, 0.5):
                c = np.clip(np.floor((x - x0) / ps + dx).astype(np.int64), 0, shape[1] - 1)
                r_ = np.clip(np.floor((ytop - y) / ps + dy).astype(np.int64), 0, shape[0] - 1)
                reg["on_occupied_if_shifted_px"][f"{dx:+},{dy:+}"] = round(
                    float(occ[r_, c].mean()), 4)
        for ch, a in arrs.items():
            v = a[row, col].astype(np.float64)
            sel = on & (v > 0)
            if sel.sum() > 100:
                reg[f"{ch}_spearman_vs_intensity"] = round(float(
                    stats.spearmanr(v[sel], it[sel]).statistic), 3)
                fit = stats.linregress(v[sel], z[sel])
                reg[f"{ch}_vs_z"] = {"slope_m_per_level": round(float(fit.slope), 5),
                                     "r2": round(float(fit.rvalue ** 2), 3)}
        out["registration"] = reg
    else:
        out["registration"] = {"points": 0}
    return out


def run_crosscheck(cfg: CrosscheckCfg, root: Path) -> tuple[dict, list[Path]]:
    by_ch = {ch: {p.stem: p for p in _glob(root, pat)} for ch, pat in cfg.channels.items()}
    common = set.intersection(*(set(v) for v in by_ch.values()))
    dens = by_ch[cfg.density_channel]
    names = sorted(common, key=lambda n: -dens[n].stat().st_size)[: cfg.n_tiles]
    point_files = [p for pat in cfg.points for p in _glob(root, pat)]
    point_tiles = []
    for p in point_files:
        with laspy.open(p) as f:
            hd = f.header
            point_tiles.append((p, (hd.mins[0], hd.mins[1], hd.maxs[0], hd.maxs[1])))
    jobs = [(n, {ch: by_ch[ch][n] for ch in by_ch}, cfg.density_channel, point_tiles,
             cfg.max_points_per_tile, cfg.seed + i) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        rows = list(ex.map(tile_check, jobs))
    rep = summarize(rows, list(by_ch))
    rep["name_vs_position"] = name_pitch(list(dens.values()))
    return rep, [by_ch[ch][n] for n in names for ch in by_ch]


def name_pitch(paths: list[Path]) -> dict:
    """Fit tile position = pitch * name index + c. Pitch and residuals only, never c."""
    ix, iy, xs, ys, w = [], [], [], [], 0.0
    for p in paths:
        idx = tile_name_index(p.name)
        if idx is None:
            continue
        with rasterio.open(p) as ds:
            t = ds.transform
            w = ds.width * t.a
        ix.append(idx[0])
        iy.append(idx[1])
        xs.append(t.c)
        ys.append(t.f)
    if len(ix) < 3:
        return {}
    out = {"tile_size_m": round(w, 3)}
    for label, a, b in (("x~first", ix, xs), ("y~second", iy, ys),
                        ("x~second", iy, xs), ("y~first", ix, ys)):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if np.ptp(a) == 0:
            continue
        fit = stats.linregress(a, b)
        res = b - (fit.slope * a + fit.intercept)
        out[label] = {"pitch_m": round(float(fit.slope), 3), "r2": round(float(fit.rvalue ** 2), 4),
                      "max_abs_residual_m": round(float(np.abs(res).max()), 3)}
    return out


def _hist_q(h: np.ndarray) -> dict:
    tot = h.sum()
    if not tot:
        return {"n": 0}
    cdf = np.cumsum(h) / tot
    q = {f"p{p * 100:g}": int(np.searchsorted(cdf, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
    return {"n": int(tot), "zero_frac": round(float(h[0] / tot), 4), **q}


def value_step(h: np.ndarray) -> int:
    """GCD of the non-zero values present: 5 means the channel is quantised in steps of 5."""
    vals = np.flatnonzero(h)
    vals = vals[vals > 0]
    return int(np.gcd.reduce(vals)) if len(vals) else 0


def summarize(rows: list[dict], channels: list[str]) -> dict:
    occ = {ch: sum(r["hist_occ"][ch] for r in rows) for ch in channels}
    emp = {ch: sum(r["hist_empty"][ch] for r in rows) for ch in channels}
    regs = [r["registration"] for r in rows if "registration" in r]
    return {
        "tiles_checked": len(rows),
        "occupied_frac": round(sum(r["occupied_px"] for r in rows) / sum(r["px"] for r in rows), 4),
        "origin_offset_px": sorted({tuple(r["origin_offset_px"]) for r in rows}),
        "name_rules_matching_all": sorted(set.intersection(*(set(r["name_rules"]) for r in rows)))
        if rows else [],
        "where_density_gt_0": {ch: {**_hist_q(occ[ch]), "value_step": value_step(occ[ch]),
                                    "distinct": int((occ[ch] > 0).sum())} for ch in channels},
        "where_density_eq_0": {ch: _hist_q(emp[ch]) for ch in channels},
        "registration_per_tile": regs,
    }
