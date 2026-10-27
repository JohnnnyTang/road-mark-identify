"""Infer what delivered raster channels encode, from the rasters and the point tiles (1.8).

Tests (all redacted: value distributions and fitted constants, no positions):
  density   value = k * count?  Per small window on covered ground, pixel counts are about
            Poisson(lambda): P(0) = exp(-lambda), mean count = lambda. So
            k = mean(value) / -ln(share of zeros). The share of values not divisible by k is
            a second check.
  intensity Per count level n = value / k: if the pixel value is the max of n points, its
            median rises with n; if it is the mean, the median stays flat and the spread
            shrinks. On single-point pixels that hold a delivered point, the raster value must
            be a function of that point's intensity (if the thinned set is a subset of the
            source points): table of raster value per point intensity, with spread.
  height    zero share and median per count level (a within-pixel z-range is 0 for one point),
            and rank correlation with each point's height above a local ground (HAG).

World files may give the pixel corner instead of the centre (``corner_world_file``).
"""

from __future__ import annotations

import glob
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
import rasterio
from pydantic import BaseModel, Field
from scipy import stats


class ProbeCfg(BaseModel):
    channels: dict[str, str]  # density / intensity / height -> tile glob
    points: list[str] = Field(default_factory=list)
    n_tiles: int = 4
    window_px: int = 10
    corner_world_file: bool = True  # Step 1.8 crosscheck: .tfw holds the corner
    hag_cell_m: float = 1.0
    seed: int = 0
    workers: int = Field(default=2, exclude=True)


def _glob(root: Path, pat: str) -> list[Path]:
    return sorted(Path(f) for f in glob.glob(str(Path(pat) if Path(pat).is_absolute()
                                                 else root / pat)))


def poisson_k(dens: np.ndarray, win: int) -> np.ndarray:
    """k = mean(value) / -ln(P0) per win x win window with 5-95 % empty pixels."""
    h, w = (dens.shape[0] // win) * win, (dens.shape[1] // win) * win
    d = dens[:h, :w].reshape(h // win, win, w // win, win).astype(np.float64)
    p0 = (d == 0).mean(axis=(1, 3))
    mean = d.mean(axis=(1, 3))
    ok = (p0 > 0.05) & (p0 < 0.95)
    return mean[ok] / -np.log(p0[ok])


def joint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """256 x 256 joint histogram of two uint8-range arrays."""
    idx = a.astype(np.int64).ravel() * 256 + np.minimum(b, 255).astype(np.int64).ravel()
    return np.bincount(idx, minlength=65536).reshape(256, 256)


def _local_hag(x, y, z, cell):
    """Height above the median z of the point's ground cell (thinned ground points only)."""
    cx = np.floor((x - x.min()) / cell).astype(np.int64)
    cy = np.floor((y - y.min()) / cell).astype(np.int64)
    key = cy * (cx.max() + 1) + cx
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], z[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    med = np.empty(len(starts))
    for i, (a, b) in enumerate(zip(starts, np.r_[starts[1:], len(ks)], strict=True)):
        med[i] = np.median(zs[a:b])
    idx = np.searchsorted(ks[starts], key)
    return z - med[idx]


def probe_tile(args) -> dict:
    name, paths, point_tiles, cfg = args
    rng = np.random.default_rng(cfg.seed)
    arrs = {}
    for ch, p in paths.items():
        with rasterio.open(p) as ds:
            arrs[ch] = ds.read(1)
            t, shape = ds.transform, (ds.height, ds.width)
    ps = t.a
    x0, ytop = (t.c + ps / 2, t.f - ps / 2) if cfg.corner_world_file else (t.c, t.f)
    w, h = shape[1] * ps, shape[0] * ps
    dens, inten, hgt = arrs["density"], arrs.get("intensity"), arrs.get("height")
    occ = dens > 0
    out = {"k": poisson_k(dens, cfg.window_px),
           "dens_hist": np.bincount(dens[occ], minlength=256)}
    if inten is not None:
        out["dens_x_int"] = joint(dens[occ], inten[occ])
    if hgt is not None:
        out["dens_x_hgt"] = joint(dens[occ], hgt[occ])

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
        its.append(np.asarray(las.intensity, np.int64)[m])
    if xs and sum(len(v) for v in xs) > 100:
        x, y, z, it = (np.concatenate(v) for v in (xs, ys, zs, its))
        hag = _local_hag(x, y, z, cfg.hag_cell_m)
        col = np.clip(np.floor((x - x0) / ps).astype(np.int64), 0, shape[1] - 1)
        row = np.clip(np.floor((ytop - y) / ps).astype(np.int64), 0, shape[0] - 1)
        dv = dens[row, col]
        out["paired_points"] = len(x)
        out["paired_dens_hist"] = np.bincount(dv, minlength=256)
        if inten is not None:
            rv = inten[row, col].astype(np.int64)
            out["pt_x_raster_int"] = {int(d): joint(np.minimum(it[dv == d], 255), rv[dv == d])
                                      for d in np.unique(dv) if d}
        if hgt is not None:
            sel = dv > 0
            if sel.sum() > 1000:
                pick = rng.choice(np.flatnonzero(sel), min(int(sel.sum()), 200_000), replace=False)
                out["hgt_hag"] = (hgt[row[pick], col[pick]].astype(np.float64), hag[pick])
    return out


def _rows_q(h2: np.ndarray, levels: list[int]) -> dict:
    """For a joint histogram (level x value): per level n, median, IQR and zero share of value."""
    out = {}
    for lv in levels:
        row = h2[lv]
        tot = row.sum()
        if tot < 100:
            continue
        cdf = np.cumsum(row) / tot
        q = [int(np.searchsorted(cdf, p)) for p in (0.25, 0.5, 0.75, 0.95)]
        out[str(lv)] = {"n": int(tot), "p25": q[0], "p50": q[1], "p75": q[2], "p95": q[3],
                        "zero_frac": round(float(row[0] / tot), 3)}
    return out


def summarize(rows: list[dict]) -> dict:
    k = np.concatenate([r["k"] for r in rows])
    dh = sum(r["dens_hist"] for r in rows)
    occ_n = dh.sum()
    k_est = float(np.median(k)) if len(k) else float("nan")
    ki = max(1, int(round(k_est)))
    rep = {
        "density": {
            "k_poisson": {"windows": int(len(k)), "median": round(k_est, 3),
                          "p25": round(float(np.quantile(k, 0.25)), 3) if len(k) else None,
                          "p75": round(float(np.quantile(k, 0.75)), 3) if len(k) else None},
            "share_not_multiple_of_k": round(float(
                dh[np.arange(256) % ki != 0].sum() / max(occ_n, 1)), 4),
            "top_values": {str(int(v)): round(float(dh[v] / occ_n), 4)
                           for v in np.argsort(dh)[::-1][:12] if dh[v]},
        },
    }
    levels = [v for v in np.argsort(dh)[::-1][:8] if dh[v]]
    levels.sort()
    if all("dens_x_int" in r for r in rows):
        rep["intensity_by_density_value"] = _rows_q(sum(r["dens_x_int"] for r in rows), levels)
    if all("dens_x_hgt" in r for r in rows):
        hj = sum(r["dens_x_hgt"] for r in rows)
        rep["height_by_density_value"] = _rows_q(hj, levels)
        hv = hj.sum(0)
        if (hv > 0).sum() <= 32:  # a small code book: list it with shares
            rep["height_values_present"] = {str(int(v)): round(float(hv[v] / hv.sum()), 4)
                                            for v in np.flatnonzero(hv)}
    paired = [r for r in rows if "paired_points" in r]
    if paired:
        pd = sum(r["paired_dens_hist"] for r in paired)
        rep["points_on_pixels"] = {
            "points": int(sum(r["paired_points"] for r in paired)),
            "share_on_empty_pixel": round(float(pd[0] / pd.sum()), 4),
            "density_value_at_points": {str(int(v)): round(float(pd[v] / pd.sum()), 4)
                                        for v in np.argsort(pd)[::-1][:6] if pd[v]}}
        tables = {}
        for r in paired:
            for d, h2 in r.get("pt_x_raster_int", {}).items():
                tables[d] = tables.get(d, 0) + h2
        rep["raster_int_vs_point_int"] = {}
        for d in sorted(tables, key=lambda d: -tables[d].sum())[:3]:
            h2 = tables[d]
            pts_levels = [int(v) for v in np.argsort(h2.sum(1))[::-1][:12] if h2[v].sum()]
            per = _rows_q(h2, sorted(pts_levels))
            # Determinism: share of pairs equal to the modal raster value of their point value.
            modal = h2.argmax(axis=1)
            det = float(h2[np.arange(256), modal].sum() / max(h2.sum(), 1))
            xs = np.repeat(np.arange(256), h2.sum(1).astype(np.int64))
            ys = np.concatenate([np.repeat(np.arange(256), h2[i].astype(np.int64))
                                 for i in range(256)])
            rho = float(stats.spearmanr(xs, ys).statistic) if len(xs) > 10 else None
            ratio = (ys[xs > 0] / xs[xs > 0]) if (xs > 0).any() else np.array([np.nan])
            rep["raster_int_vs_point_int"][f"density={d}"] = {
                "pairs": int(h2.sum()), "spearman": None if rho is None else round(rho, 3),
                "share_at_modal_value": round(det, 3),
                "ratio_raster_over_point": {"p25": round(float(np.quantile(ratio, 0.25)), 3),
                                            "p50": round(float(np.median(ratio)), 3),
                                            "p75": round(float(np.quantile(ratio, 0.75)), 3)},
                "raster_value_per_point_value": per}
        hh = [r["hgt_hag"] for r in paired if "hgt_hag" in r]
        if hh:
            hv = np.concatenate([a for a, _ in hh])
            hg = np.concatenate([b for _, b in hh])
            bins = np.quantile(hg, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
            which = np.clip(np.searchsorted(bins, hg, side="right") - 1, 0, 5)
            rep["height_vs_point_hag"] = {
                "spearman": round(float(stats.spearmanr(hv, hg).statistic), 3),
                "by_hag_bin": [{"hag_cm": [round(bins[i] * 100, 1), round(bins[i + 1] * 100, 1)],
                                "height_zero_frac": round(float((hv[which == i] == 0).mean()), 3),
                                "height_median": float(np.median(hv[which == i]))}
                               for i in range(6) if (which == i).any()]}
    return rep


def run_probe(cfg: ProbeCfg, root: Path) -> tuple[dict, list[Path]]:
    by_ch = {ch: {p.stem: p for p in _glob(root, pat)} for ch, pat in cfg.channels.items()}
    common = set.intersection(*(set(v) for v in by_ch.values()))
    dens = by_ch["density"]
    names = sorted(common, key=lambda n: -dens[n].stat().st_size)[: cfg.n_tiles]
    point_tiles = []
    for pat in cfg.points:
        for p in _glob(root, pat):
            with laspy.open(p) as f:
                hd = f.header
                point_tiles.append((p, (hd.mins[0], hd.mins[1], hd.maxs[0], hd.maxs[1])))
    jobs = [(n, {ch: by_ch[ch][n] for ch in by_ch}, point_tiles,
             cfg.model_copy(update={"seed": cfg.seed + i})) for i, n in enumerate(names)]
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        rows = list(ex.map(probe_tile, jobs))
    return summarize(rows), [by_ch[ch][n] for n in names for ch in by_ch]
