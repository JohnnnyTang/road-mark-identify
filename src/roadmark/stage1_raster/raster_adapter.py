"""Delivered raster tiles -> I1 channel rasters (Roadmap 1.8; DECISIONS 2026-10-26, Step 1.8).

The delivery has no point cloud to rasterize, only uint8 tiles per channel. This adapter
produces the same file set as Stage 1 (``rasterize.CHANNELS``), on one grid, as COGs:

    grid        world files hold the pixel corner (``corner_world_file``): shift by half a
                pixel, then every tile sits on one integer pixel lattice (checked, never rounded
                silently)
    decode      density   = floor(k * min(count, sat))        -> count (uint16)
                intensity = floor(255 * min(I_max, sat) / sat) -> int_max in point units
                density = 0 is no data: int_max NaN, coverage 0
    overlaps    each pixel comes from the tile whose centre is nearest (on a regular tile
                lattice: the tile's core). The overlaps are compared first and reported.
    aggregate   optional k x k: count = sum, int_max = max (exact for a max channel)

int_mean, int_p90, hag_max and ground_z can't be recovered from the delivery. They are written
all-NaN (no data) so consumers find the full I1 set, and the manifest says so.

The mosaic is a single raster, so its bounding box must stay moderate (``max_grid_mpx``):
select the pilot tiles by glob. TODO(scale): per-tile COGs + VRT for the full area.

Redacted report: counts, shares and pixel sizes only. No origins, extents or tile names.
"""

from __future__ import annotations

import glob
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from pydantic import BaseModel
from rasterio.windows import Window

from roadmark.io.raster import Grid, create_tiled, to_cog, write_window
from roadmark.io.raster_inventory import tile_name_index
from roadmark.stage1_raster.rasterize import CHANNELS

log = logging.getLogger(__name__)

SEMANTICS = {
    "density": "ground point count per pixel, decoded from the delivered density channel; "
               "the saturated value is a lower bound",
    "coverage": "1 where density >= coverage_min_points",
    "int_max": "max point intensity per pixel (8-bit LAS scale), decoded from the delivered "
               "intensity channel; not range/incidence corrected; the saturated value is a "
               "lower bound",
    "int_mean": "unavailable: the delivery holds a per-pixel max only (all NaN)",
    "int_p90": "unavailable: the delivery holds a per-pixel max only (all NaN)",
    "hag_max": "unavailable: the delivered height channel is not height above ground (all NaN)",
    "ground_z": "unavailable: the delivered height channel is not elevation (all NaN)",
}


class AdapterCfg(BaseModel):
    channels: dict[str, list[str]]  # "density" / "intensity" -> globs relative to the data root
    corner_world_file: bool = True  # Step 1.8 crosscheck: the .tfw holds the pixel corner
    density_per_point: float = 5.1
    density_saturation: int = 50
    intensity_saturation: float = 45.0
    aggregate: int = 1
    coverage_min_points: int = 1  # ~1 point per 3 cm pixel: one point is evidence
    max_grid_mpx: float = 1500.0
    block_px: int = 4096
    compress: str = "DEFLATE"
    tiles: list[str] | None = None  # tile names (stems) to use; None = everything globbed


@dataclass
class _Tile:
    stem: str
    paths: dict[str, Path]
    row: int  # offset in the native mosaic (px)
    col: int
    height: int
    width: int

    @property
    def centre(self) -> tuple[float, float]:
        return self.row + self.height / 2, self.col + self.width / 2

    def rect(self) -> tuple[int, int, int, int]:
        return self.row, self.col, self.row + self.height, self.col + self.width


def density_lut(cfg: AdapterCfg) -> np.ndarray:
    v = np.arange(256, dtype=np.float64)
    return np.minimum(np.rint(v / cfg.density_per_point), cfg.density_saturation).astype(np.uint16)


def intensity_lut(cfg: AdapterCfg) -> np.ndarray:
    v = np.arange(256, dtype=np.float64)
    return np.rint(v * cfg.intensity_saturation / 255.0).astype(np.float32)


def _glob(root: Path, pats: list[str]) -> list[Path]:
    return sorted({Path(f) for p in pats for f in glob.glob(
        str(Path(p) if Path(p).is_absolute() else root / p))})


def _lattice(path: Path, corner: bool) -> tuple[float, int, int, int, int, object]:
    """Pixel size, absolute lattice row/col of the top-left pixel, height, width, CRS."""
    with rasterio.open(path) as ds:
        t = ds.transform
        if ds.count != 1 or ds.dtypes[0] != "uint8":
            raise ValueError(f"{path.name}: expected one uint8 band")
        if t.b or t.d or not math.isclose(t.a, -t.e, rel_tol=1e-9):
            raise ValueError(f"{path.name}: rotated or non-square pixels")
        ps = t.a
        x0, ytop = (t.c + ps / 2, t.f - ps / 2) if corner else (t.c, t.f)
        fc, fr = x0 / ps, -ytop / ps
        if abs(fc - round(fc)) > 1e-3 or abs(fr - round(fr)) > 1e-3:
            raise ValueError(f"{path.name}: origin is not on the pixel lattice "
                             "(check corner_world_file)")
        return ps, round(fr), round(fc), ds.height, ds.width, ds.crs


def collect_tiles(cfg: AdapterCfg, root: Path) -> tuple[list[_Tile], Grid, object]:
    if set(cfg.channels) != {"density", "intensity"}:
        raise ValueError("channels must be exactly density and intensity "
                         "(the delivered height channel is not used)")
    by_ch: dict[str, dict[str, Path]] = {}
    for ch, pats in cfg.channels.items():
        files = _glob(root, pats)
        stems = [p.stem for p in files]
        if len(set(stems)) != len(stems):
            raise ValueError(f"{ch}: duplicate tile names across the globs")
        by_ch[ch] = dict(zip(stems, files, strict=True))
    common = sorted(by_ch["density"].keys() & by_ch["intensity"].keys())
    missing = len(by_ch["density"].keys() ^ by_ch["intensity"].keys())
    if cfg.tiles is not None:
        unknown = set(cfg.tiles) - set(common)
        if unknown:
            raise FileNotFoundError(f"{len(unknown)} selected tiles lack density or intensity")
        common, missing = sorted(set(cfg.tiles)), 0
    if not common:
        raise FileNotFoundError("no tile has both a density and an intensity raster")
    if missing:
        log.warning("%d tiles have only one channel and are skipped", missing)

    f = cfg.aggregate
    lat, ps, crs = {}, None, None
    for s in common:
        g = {ch: _lattice(by_ch[ch][s], cfg.corner_world_file) for ch in by_ch}
        if g["density"][:5] != g["intensity"][:5]:
            raise ValueError(f"{s}: density and intensity grids differ")
        tps, r, c, h, w, tcrs = g["density"]
        if ps is None:
            ps, crs = tps, tcrs
        elif not math.isclose(tps, ps, rel_tol=1e-9) or tcrs != crs:
            raise ValueError(f"{s}: pixel size or CRS differs from the other tiles")
        lat[s] = (r, c, h, w)
    r0 = min(v[0] for v in lat.values()) // f * f
    c0 = min(v[1] for v in lat.values()) // f * f
    r1 = -(-max(v[0] + v[2] for v in lat.values()) // f) * f
    c1 = -(-max(v[1] + v[3] for v in lat.values()) // f) * f
    if (r1 - r0) * (c1 - c0) / 1e6 > cfg.max_grid_mpx:
        raise ValueError(f"mosaic would be {(r1 - r0) * (c1 - c0) / 1e6:.0f} Mpx "
                         f"(> max_grid_mpx {cfg.max_grid_mpx:g}): select fewer tiles")
    tiles = [_Tile(s, {ch: by_ch[ch][s] for ch in by_ch}, r - r0, c - c0, h, w)
             for s, (r, c, h, w) in lat.items()]
    grid = Grid(c0 * ps, -r0 * ps, ps, c1 - c0, r1 - r0)
    return tiles, grid, crs


def _overlap(a: _Tile, b: _Tile) -> tuple[int, int, int, int] | None:
    ar0, ac0, ar1, ac1 = a.rect()
    br0, bc0, br1, bc1 = b.rect()
    r0, c0, r1, c1 = max(ar0, br0), max(ac0, bc0), min(ar1, br1), min(ac1, bc1)
    return (r0, c0, r1, c1) if r0 < r1 and c0 < c1 else None


def _d2(t: _Tile, r0: int, c0: int, r1: int, c1: int) -> np.ndarray:
    """Squared distance of pixel centres in the window to the tile centre."""
    tr, tc = t.centre
    rr = (np.arange(r0, r1, dtype=np.float64) + 0.5 - tr) ** 2
    cc = (np.arange(c0, c1, dtype=np.float64) + 0.5 - tc) ** 2
    return rr[:, None] + cc[None, :]


def _read(t: _Tile, ch: str, win: tuple[int, int, int, int] | None = None) -> np.ndarray:
    with rasterio.open(t.paths[ch]) as ds:
        if win is None:
            return ds.read(1)
        r0, c0, r1, c1 = win
        return ds.read(1, window=Window(c0 - t.col, r0 - t.row, c1 - c0, r1 - r0))


def _local(t: _Tile, win: tuple[int, int, int, int]) -> tuple[slice, slice]:
    r0, c0, r1, c1 = win
    return slice(r0 - t.row, r1 - t.row), slice(c0 - t.col, c1 - t.col)


class _OverlapStats:
    """Do neighbouring tiles agree where they overlap? Totals plus the worst pair."""

    def __init__(self) -> None:
        self.tot = dict.fromkeys(("px", "either", "both", "density_equal", "intensity_equal",
                                  "owner_empty_other_occupied"), 0)
        self.pairs: list[dict[str, float | None]] = []
        self.n_pairs = 0

    def add(self, da: np.ndarray, db: np.ndarray, ia: np.ndarray, ib: np.ndarray,
            a_owns: np.ndarray) -> None:
        oa, ob = da > 0, db > 0
        either, both = oa | ob, oa & ob
        n = {"px": da.size, "either": int(either.sum()), "both": int(both.sum()),
             "density_equal": int((either & (da == db)).sum()),
             "intensity_equal": int((both & (ia == ib)).sum()),
             "owner_empty_other_occupied": int(np.where(a_owns, ~oa & ob, ~ob & oa).sum())}
        for k, v in n.items():
            self.tot[k] += v
        self.n_pairs += 1
        if n["either"]:  # an empty overlap has nothing to agree on
            self.pairs.append(self._shares(n))

    @staticmethod
    def _shares(n: dict[str, int]) -> dict[str, float | None]:
        e, b = n["either"], n["both"]
        return {"both_occupied_of_either": round(b / e, 4) if e else None,
                "density_equal_of_either": round(n["density_equal"] / e, 4) if e else None,
                "intensity_equal_of_both": round(n["intensity_equal"] / b, 4) if b else None,
                "owner_empty_other_occupied_of_either":
                    round(n["owner_empty_other_occupied"] / e, 4) if e else None}

    def report(self) -> dict:
        out: dict = {"pairs": self.n_pairs, "pairs_with_points": len(self.pairs),
                     "px": self.tot["px"], "occupied_in_either": self.tot["either"]}
        if self.pairs:
            out["all"] = self._shares(self.tot)
            worst = {}
            for k in self.pairs[0]:
                vals = [p[k] for p in self.pairs if p[k] is not None]
                pick = max if k.startswith("owner_empty") else min
                worst[k] = pick(vals) if vals else None
            out["worst_pair"] = worst
        return out


def mosaic(tiles: list[_Tile], grid: Grid, crs, work: Path, cfg: AdapterCfg) -> dict:
    """Decode every tile into the native mosaic (count, int_max) by nearest-centre ownership."""
    dlut, ilut = density_lut(cfg), intensity_lut(cfg)
    out = {"density": work / "native_density.tif", "int_max": work / "native_int_max.tif"}
    create_tiled(out["density"], grid, "uint16", None, crs, "density", sparse=True)
    create_tiled(out["int_max"], grid, "float32", float("nan"), crs, "int_max", sparse=True)
    stats = _OverlapStats()
    for i, t in enumerate(tiles):
        dv, iv = _read(t, "density"), _read(t, "intensity")
        owned = np.ones(dv.shape, bool)
        for j, n in enumerate(tiles):
            win = None if j == i else _overlap(t, n)
            if win is None:
                continue
            dt, dn = _d2(t, *win), _d2(n, *win)
            t_owns = (dt < dn) | ((dt == dn) & (i < j))
            sl = _local(t, win)
            owned[sl] &= t_owns
            if i < j:  # compare each overlapping pair once
                stats.add(dlut[dv[sl]], dlut[_read(n, "density", win)],
                          iv[sl], _read(n, "intensity", win), t_owns)
        rows, cols = np.flatnonzero(owned.any(1)), np.flatnonzero(owned.any(0))
        if not len(rows):
            continue
        sr, sc = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
        own = owned[sr, sc]
        cnt = dlut[dv[sr, sc]]
        imax = np.where(cnt > 0, ilut[iv[sr, sc]], np.float32(np.nan))
        w = Window(t.col + sc.start, t.row + sr.start, own.shape[1], own.shape[0])
        for k, new in (("density", cnt), ("int_max", imax)):
            with rasterio.open(out[k], "r+") as ds:
                old = ds.read(1, window=w)
                ds.write(np.where(own, new, old), 1, window=w)
    return {"paths": out, "overlaps": stats.report()}


def finalize(native: dict[str, Path], ngrid: Grid, crs, work: Path, cfg: AdapterCfg
             ) -> tuple[dict[str, Path], Grid, dict]:
    """Aggregate k x k (or copy), derive coverage, write the I1 channel set."""
    f = cfg.aggregate
    grid = Grid(ngrid.x0, ngrid.y0, ngrid.pixel_size * f, ngrid.width // f, ngrid.height // f)
    tmp = {k: work / f"{k}.tif" for k in CHANNELS}
    for k, (dtype, nodata) in CHANNELS.items():
        create_tiled(tmp[k], grid, dtype, nodata, crs, k, sparse=True)
    hist = np.zeros(cfg.density_saturation * f * f + 2, np.int64)
    sat_int = 0
    with rasterio.open(native["density"]) as dsd, rasterio.open(native["int_max"]) as dsi:
        for w in grid.blocks(cfg.block_px):
            nw = Window(w.col_off * f, w.row_off * f, w.width * f, w.height * f)
            cnt, imax = dsd.read(1, window=nw).astype(np.uint32), dsi.read(1, window=nw)
            if f > 1:
                h, wd = int(w.height), int(w.width)
                cnt = cnt.reshape(h, f, wd, f).sum(axis=(1, 3))
                imax = np.fmax.reduce(np.fmax.reduce(imax.reshape(h, f, wd, f), axis=3), axis=1)
            if not cnt.any():
                continue
            hist += np.bincount(np.minimum(cnt.ravel(), len(hist) - 1), minlength=len(hist))
            sat_int += int((imax >= cfg.intensity_saturation).sum())
            write_window(tmp["density"], np.minimum(cnt, np.iinfo(np.uint16).max)
                         .astype(np.uint16), w)
            write_window(tmp["coverage"], (cnt >= cfg.coverage_min_points).astype(np.uint8), w)
            write_window(tmp["int_max"], imax.astype(np.float32), w)
    occ = int(hist[1:].sum())
    cdf = np.cumsum(hist[1:]) / max(1, occ)
    summary = {
        "occupied_px": occ,
        "occupied_frac_of_grid": round(occ / (grid.width * grid.height), 4),
        "covered_px": int(hist[cfg.coverage_min_points:].sum()),
        "points_per_occupied_px": {f"p{q * 100:g}": int(np.searchsorted(cdf, q)) + 1
                                   for q in (0.25, 0.5, 0.75, 0.95)} if occ else {},
        "single_point_frac_of_occupied": round(float(hist[1] / max(1, occ)), 4),
        "density_saturated_frac_of_occupied":
            round(float(hist[cfg.density_saturation:].sum() / max(1, occ)), 4) if f == 1 else None,
        "intensity_saturated_frac_of_occupied": round(sat_int / max(1, occ), 4),
    }
    return tmp, grid, summary


def run_adapter(cfg: AdapterCfg, root: Path, run_dir: Path) -> tuple[dict, list[Path]]:
    if cfg.aggregate < 1:
        raise ValueError("aggregate must be >= 1")
    tiles, ngrid, crs = collect_tiles(cfg, root)
    log.info("%d tiles, native mosaic %dx%d px", len(tiles), ngrid.width, ngrid.height)
    work = run_dir / "work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    mos = mosaic(tiles, ngrid, crs, work, cfg)
    tmp, grid, summary = finalize(mos["paths"], ngrid, crs, work, cfg)
    for k, src in tmp.items():
        to_cog(src, run_dir / f"{k}.tif", cfg.compress, "nearest" if k == "coverage" else "average")
    shutil.rmtree(work, ignore_errors=True)
    report = {
        "grid": {"width_px": grid.width, "height_px": grid.height,
                 "pixel_size": round(grid.pixel_size, 6), "aggregate": cfg.aggregate,
                 "native_pixel_size": round(ngrid.pixel_size, 6),
                 "epsg": crs.to_epsg() if crs else None},
        "tiles": len(tiles),
        "overlaps": mos["overlaps"],
        "coverage": summary,
        "channels": SEMANTICS,
        "outputs": sorted(f"{k}.tif" for k in tmp),
    }
    inputs = [p for t in tiles for p in t.paths.values()]
    return report, inputs


def _occupied(path: Path) -> int:
    with rasterio.open(path) as ds:
        return int(np.count_nonzero(ds.read(1)))


def pick_pilot(density: list[Path], block: int = 2, workers: int = 4) -> tuple[list[str], dict]:
    """The block x block group of neighbouring tiles (by name index) with the most occupied
    pixels: a contiguous pilot area that also exercises edge and corner overlaps.

    Returns the tile names (keep them in the data root) and a redacted summary.
    """
    from concurrent.futures import ProcessPoolExecutor

    idx = {tile_name_index(p.name): p for p in density}
    idx.pop(None, None)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        occ = dict(zip(idx, ex.map(_occupied, idx.values()), strict=True))
    groups = []
    for i, j in idx:
        members = [(i + a, j + b) for a in range(block) for b in range(block)]
        if all(m in idx for m in members):
            groups.append((sum(occ[m] for m in members), members))
    if not groups:
        raise ValueError(f"no complete {block}x{block} group of neighbouring tiles")
    groups.sort(key=lambda g: -g[0])
    best, members = groups[0]
    ranked = sorted(occ.values(), reverse=True)
    summary = {"tiles_scanned": len(idx), "groups_considered": len(groups),
               "picked_tiles": len(members), "picked_occupied_px": best,
               "picked_tile_ranks_by_occupancy": sorted(ranked.index(occ[m]) + 1
                                                        for m in members),
               "picked_vs_median_group": round(best / max(1, groups[len(groups) // 2][0]), 2)}
    return [idx[m].stem for m in members], summary
