"""Stage 1 driver: LAS/LAZ files -> channel COGs + road mask + QA report.

    A  partition    stream every input once, write per-block shards (block + halo)
    B  geometry     per block: ground classification, height above ground, scanner geometry;
                    geometry channels (density, coverage, hag_max, ground_z)
    C  road mask    curb steps -> road-surface prior (needs geometry only)
    D  intensity    fit the range/incidence model + per-pass normalisation on ground points
                    *inside the road mask* (global, never per block or per tile)
    E  rasterize    per block: correct intensity, intensity channels of the core window
    F  finalize     COG conversion, QA report, manifest

TODO(scale): C holds the full-resolution rasters in memory. For areas beyond a few km of
road, compute the road mask on a coarser grid (curb steps don't need 2.5 cm).

Everything is written under ``<data_root>/runs/<name>/stage1/``: outside git by construction.
"""

from __future__ import annotations

import glob
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window
from scipy import ndimage

from roadmark.config import PipelineConfig
from roadmark.io.pointcloud import iter_chunks, read_header
from roadmark.io.raster import Grid, create_tiled, to_cog, write_window
from roadmark.provenance import write_manifest
from roadmark.stage1_raster import ground as G
from roadmark.stage1_raster import intensity as I
from roadmark.stage1_raster import qa
from roadmark.stage1_raster.rasterize import CHANNELS, cell_stats, hag_max
from roadmark.stage1_raster.road_mask import road_surface

log = logging.getLogger(__name__)
QA_PAINT_RATIO = 2.5  # QA only: paint cells = corrected int_mean > ratio * asphalt median


def resolve_inputs(cfg: PipelineConfig) -> list[Path]:
    root = cfg.resolved_data_root()
    files: list[Path] = []
    for pat in cfg.stage1.inputs:
        p = Path(pat) if Path(pat).is_absolute() else root / pat
        files += [Path(f) for f in sorted(glob.glob(str(p)))]
    if not files:
        raise FileNotFoundError("stage1.inputs matched no files under the data root")
    return files


def load_trajectory(cfg: PipelineConfig) -> dict[str, np.ndarray] | None:
    tcfg = cfg.stage1.trajectory
    if tcfg.path is None:
        return None
    p = cfg.resolved_data_root() / tcfg.path
    if p.suffix == ".npz":
        d = np.load(p)
        traj = {k: d[k].astype(np.float64) for k in ("t", "x", "y", "z")}
    else:
        arr = np.loadtxt(p, delimiter=",", skiprows=1)
        traj = dict(zip(("t", "x", "y", "z"), arr[:, :4].T, strict=True))
    order = np.argsort(traj["t"])
    return {k: v[order] for k, v in traj.items()}


def _check_crs(cfg: PipelineConfig, headers) -> CRS | None:
    if cfg.crs_epsg is None:
        return None
    for h in headers:
        if h.epsg is not None and h.epsg != cfg.crs_epsg:
            raise ValueError(f"{h.path.name}: EPSG {h.epsg} != configured {cfg.crs_epsg}")
        if h.crs_is_projected is False:
            raise ValueError(f"{h.path.name}: CRS is not projected; reproject first (§2.5)")
    return CRS.from_epsg(cfg.crs_epsg)


def _load_shards(paths: list[Path]) -> dict[str, np.ndarray]:
    parts = [dict(np.load(p)) for p in paths]
    return {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}


def run_stage1(cfg: PipelineConfig, keep_work: bool = False) -> dict:
    s1 = cfg.stage1
    rc = s1.raster
    inputs = resolve_inputs(cfg)
    headers = [read_header(p) for p in inputs]
    crs = _check_crs(cfg, headers)
    run_dir = cfg.resolved_data_root() / "runs" / cfg.name / "stage1"
    work = run_dir / "work"
    shutil.rmtree(work, ignore_errors=True)
    (work / "parts").mkdir(parents=True)

    b = np.array([h.bounds for h in headers])
    grid = Grid.from_bounds(b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max(),
                            rc.pixel_size)
    blocks = grid.blocks(rc.block_size_px)
    bbounds = [grid.window_bounds(w) for w in blocks]
    log.info("grid %dx%d px, %d blocks", grid.width, grid.height, len(blocks))

    # --- A: partition -----------------------------------------------------------------
    shards: dict[int, list[Path]] = {i: [] for i in range(len(blocks))}
    h = rc.halo_m
    for fi, path in enumerate(inputs):
        for ci, pts in enumerate(iter_chunks(path, s1.chunk_points)):
            if s1.pass_id_field in pts:
                pts["pass_id"] = pts.pop(s1.pass_id_field).astype(np.int32)
            x, y = pts["x"], pts["y"]
            for bi, (x0, y0, x1, y1) in enumerate(bbounds):
                m = (x >= x0 - h) & (x < x1 + h) & (y >= y0 - h) & (y < y1 + h)
                if m.any():
                    sp = work / "parts" / f"b{bi:05d}_f{fi:03d}_c{ci:05d}.npz"
                    np.savez(sp, **{k: v[m] for k, v in pts.items()})
                    shards[bi].append(sp)

    # --- B: ground, HAG, scanner geometry, geometry channels ------------------------------
    traj = load_trajectory(cfg)
    ic = s1.intensity
    tmp = {k: work / f"{k}.tif" for k in CHANNELS}
    for k, (dtype, nodata) in CHANNELS.items():
        create_tiled(tmp[k], grid, dtype, nodata, crs, k)
    low_max = s1.ground.low_nonground_max_hag
    block_files: dict[int, Path] = {}
    for bi, parts in shards.items():
        if not parts:
            continue
        pts = _load_shards(parts)
        gmask = G.classify_ground(pts, s1.ground)
        pts["is_ground"] = gmask
        pts["hag"] = G.height_above_ground(pts, gmask, s1.ground.dem_cell)
        if ic.enabled:
            pts["range"], pts["cos_inc"] = I.scanner_geometry(
                pts, traj, ic.min_cos_incidence, s1.trajectory.time_offset)
        w: Window = blocks[bi]
        shape = (int(w.height), int(w.width))
        row, col, ok = grid.cell_index(pts["x"], pts["y"], w)
        gsel = ok & pts["is_ground"]
        ch = cell_stats(row[gsel], col[gsel], shape, pts["intensity"][gsel], pts["z"][gsel])
        geo = {"density": ch["density"], "ground_z": ch["ground_z"],
               "coverage": (ch["density"] >= rc.coverage_min_points).astype(np.uint8)}
        lsel = ok & ~gmask & (pts["hag"] > 0) & (pts["hag"] <= low_max)
        geo["hag_max"] = hag_max(row[lsel], col[lsel], shape, pts["hag"][lsel], ch["density"])
        for k, arr in geo.items():
            write_window(tmp[k], arr, w)
        block_files[bi] = work / f"block_{bi:05d}.npz"
        np.savez(block_files[bi], **pts)
        for p in parts:
            p.unlink()

    # --- C: road-surface prior ----------------------------------------------------------
    with rasterio.open(tmp["ground_z"]) as a, rasterio.open(tmp["hag_max"]) as b_, \
            rasterio.open(tmp["coverage"]) as c, rasterio.open(tmp["density"]) as d:
        gz, hm, cov, den = a.read(1), b_.read(1), c.read(1), d.read(1)
        profile = c.profile
    road, curb = road_surface(gz, hm, cov, rc.pixel_size, s1.road_mask)
    del gz, hm
    profile.update(nodata=None)
    for name, arr in (("road_mask", road), ("curb", curb)):
        tmp[name] = work / f"{name}.tif"
        with rasterio.open(tmp[name], "w", **profile) as dst:
            dst.write(arr.astype(np.uint8), 1)

    # --- D: intensity model on in-road ground points -----------------------------------
    model, asphalt_med = None, float("nan")
    if ic.enabled:
        rng_gen = np.random.default_rng(cfg.seed)
        per_block = max(1, ic.fit_sample_size // max(1, len(block_files)))
        sample: dict[str, list[np.ndarray]] = {"i": [], "r": [], "c": [], "p": []}
        for bf in block_files.values():
            pts = dict(np.load(bf))
            row, col, ok = grid.cell_index(pts["x"], pts["y"])  # full-grid index
            sel = ok & pts["is_ground"]
            sel[sel] = road[row[sel], col[sel]]  # core-window points only: no halo duplicates
            idx = np.flatnonzero(sel)
            if len(idx) > per_block:
                idx = rng_gen.choice(idx, per_block, replace=False)
            sample["i"].append(pts["intensity"][idx])
            sample["r"].append(pts["range"][idx])
            sample["c"].append(pts["cos_inc"][idx])
            sample["p"].append(pts.get("pass_id", np.zeros(len(pts["x"]), np.int32))[idx])
        si, sr, sc, sp = (np.concatenate(sample[k]) for k in ("i", "r", "c", "p"))
        model = I.fit_model(si, sr, sc, sp, ic, cfg.seed)
        asphalt_med = model.asphalt_median
        log.info("intensity model a=%.3f b=%.3f identifiable=%s", model.a, model.b,
                 model.identifiable)

    # --- E: intensity channels ----------------------------------------------------------
    pair_diffs: dict[str, list[np.ndarray]] = {}
    for bi, bf in block_files.items():
        w = blocks[bi]
        shape = (int(w.height), int(w.width))
        pts = dict(np.load(bf))
        inten = pts["intensity"].astype(np.float32)
        pid = pts.get("pass_id")
        if model is not None:
            inten = model.correct(inten, pts["range"], pts["cos_inc"], pid)
        row, col, ok = grid.cell_index(pts["x"], pts["y"], w)
        gsel = ok & pts["is_ground"]
        ch = cell_stats(row[gsel], col[gsel], shape, inten[gsel], pts["z"][gsel], rc.p_quantile)
        for k in ("int_max", "int_mean", "int_p90"):
            write_window(tmp[k], ch[k], w)

        if model is not None and pid is not None and len(np.unique(pid[gsel])) > 1:
            # Interior of paint only: edge pixels mix paint and asphalt differently per pass.
            paint = ndimage.binary_erosion(ch["int_mean"] > QA_PAINT_RATIO * asphalt_med)
            by_pass = {}
            for p in np.unique(pid[gsel]):
                sel = gsel & (pid == p)
                by_pass[int(p)] = cell_stats(row[sel], col[sel], shape, inten[sel],
                                             pts["z"][sel])["int_mean"]
            for k, v in qa.pair_rel_diffs(by_pass, paint).items():
                pair_diffs.setdefault(k, []).append(v)

    # --- F: finalize --------------------------------------------------------------------
    out_paths = {}
    categorical = {"coverage", "road_mask", "curb"}
    for k, src in tmp.items():
        out_paths[k] = run_dir / f"{k}.tif"
        to_cog(src, out_paths[k], rc.compress, "nearest" if k in categorical else "average")

    report = {
        "grid": {"width_px": grid.width, "height_px": grid.height, "pixel_size": rc.pixel_size,
                 "blocks": len(blocks), "epsg": cfg.crs_epsg},
        "inputs": [hd.summary() for hd in headers],
        "intensity_model": model.to_dict() if model else None,
        "asphalt_median_corrected": asphalt_med,
        "cross_pass_consistency": qa.summarize_pairs(pair_diffs, ic.consistency_tol),
        "coverage": qa.coverage_report(den, cov, road),
        "outputs": sorted(p.name for p in out_paths.values()),
    }
    (run_dir / "qa_stage1.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_manifest(run_dir, cfg, inputs, {"stage": "stage1"})
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return report
