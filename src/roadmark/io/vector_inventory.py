"""Inventory of delivered vector annotations (I3 intake): can they become training labels?

Answers, without printing coordinates, names or free text:
  - formats, layers, geometry types, CRS presence, attribute schema (class-like values only)
  - sizes: line lengths and vertex spacing; polygon areas, side lengths, and whether polygons
    are axis-aligned or rotated boxes
  - registration with the rasters: share of features on the delivered tiles and on a grid
    (e.g. the M1 pilot), and the int_max profile across lines (a peak at offset 0 means the
    lines sit on paint centres; its width estimates the paint width)

Files and layers are reported by index (``#1``); their names go only to the report in the
data root, because file and layer names can carry place names.
"""

from __future__ import annotations

import glob
import hashlib
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pyogrio
import rasterio
import shapely
from pydantic import BaseModel, Field
from rasterio import features
from rasterio.windows import Window, from_bounds

VECTOR_EXT = {".gpkg", ".shp", ".geojson", ".json", ".kml", ".dxf", ".gml", ".tab", ".mif",
              ".sqlite", ".fgb"}
CODE_LIKE = re.compile(r"^[A-Za-z0-9_\-. ]{1,32}$")
NAME_LIKE = re.compile(r"name|road|street|addr|place|loc|city|dist|remark|note|comment|desc|"
                       r"名|路|街|地|备注|说明", re.I)
QS = (0.05, 0.25, 0.5, 0.75, 0.95)


class VectorInventoryCfg(BaseModel):
    paths: list[str]  # globs relative to the data root (recursive ** allowed)
    tiles: list[str] = Field(default_factory=list)  # raster tile globs: delivered footprint
    grid: str | None = None  # a run's stage1 folder: registration against int_max/density
    corner_world_file: bool = True
    profile_half_width_m: float = 0.6
    profile_step_m: float = 0.03
    max_values: int = 30  # print value counts for code-like fields up to this many values
    group_by: list[str] = Field(default_factory=list)  # code fields: one profile per value


def _q(v: np.ndarray, nd: int = 3) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), **{f"p{q * 100:g}": round(float(np.quantile(v, q)), nd)
                                 for q in QS}}


def _files(root: Path, pats: list[str]) -> list[Path]:
    out = set()
    for p in pats:
        for f in glob.glob(str(Path(p) if Path(p).is_absolute() else root / p), recursive=True):
            if Path(f).is_file() and Path(f).suffix.lower() in VECTOR_EXT:
                out.add(Path(f))
    return sorted(out)


def _fields(df, max_values: int) -> list[dict]:
    out = []
    for c in df.columns:
        if c == "geometry":
            continue
        s = df[c]
        info = {"field": c if not NAME_LIKE.search(c) else "(name-like field)",
                "dtype": str(s.dtype), "null": int(s.isna().sum()),
                "distinct": int(s.nunique(dropna=True))}
        vals = s.dropna().astype(str)
        if (not NAME_LIKE.search(c) and 0 < info["distinct"] <= max_values
                and vals.map(lambda v: bool(CODE_LIKE.match(v))).all()):
            info["values"] = dict(Counter(vals).most_common())
        elif s.dtype.kind in "if":
            info["quantiles"] = _q(s.to_numpy())
        out.append(info)
    return out


def _line_stats(geoms: np.ndarray) -> dict:
    parts = shapely.get_parts(geoms)
    lengths = shapely.length(parts)
    nverts = shapely.get_num_coordinates(parts)
    seg = lengths / np.maximum(nverts - 1, 1)
    closed = shapely.is_closed(parts)
    return {"parts": int(len(parts)), "multipart_features": int((shapely.get_num_geometries(
                geoms) > 1).sum()),
            "total_length_m": round(float(lengths.sum()), 1), "length_m": _q(lengths, 2),
            "mean_vertex_spacing_m": _q(seg, 3), "closed": int(closed.sum()),
            "two_vertex_lines": int((nverts == 2).sum())}


def _polygon_stats(geoms: np.ndarray) -> dict:
    parts = shapely.get_parts(geoms)
    area = shapely.area(parts)
    env = shapely.area(shapely.envelope(parts))
    mrr = shapely.oriented_envelope(parts)
    rect = shapely.area(mrr)
    ring = shapely.get_exterior_ring(mrr)
    c = shapely.get_coordinates(ring).reshape(len(parts), 5, 2)
    s1 = np.hypot(*(c[:, 1] - c[:, 0]).T)
    s2 = np.hypot(*(c[:, 2] - c[:, 1]).T)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"parts": int(len(parts)), "area_m2": _q(area, 2),
                "long_side_m": _q(np.maximum(s1, s2), 2),
                "short_side_m": _q(np.minimum(s1, s2), 2),
                "vertices": _q(shapely.get_num_coordinates(parts), 0),
                "rectangular": int((area / rect > 0.98).sum()),  # area fills its min box
                "axis_aligned_box": int((area / env > 0.98).sum()),
                "holes": int((shapely.get_num_interior_rings(parts) > 0).sum())}


def tile_rects(root: Path, pats: list[str], corner: bool) -> list[tuple[float, ...]]:
    rects = []
    for p in sorted({f for pat in pats for f in glob.glob(str(root / pat))}):
        with rasterio.open(p) as ds:
            t = ds.transform
            ps = t.a
            x0, y1 = (t.c + ps / 2, t.f - ps / 2) if corner else (t.c, t.f)
            rects.append((x0, y1 - ds.height * ps, x0 + ds.width * ps, y1))
    return rects


def _profile(lines: np.ndarray, int_ds, cfg: VectorInventoryCfg) -> dict:
    """int_max sampled across lines at signed lateral offsets (median over samples)."""
    step = cfg.profile_step_m
    offs = np.round(np.arange(-cfg.profile_half_width_m, cfg.profile_half_width_m + 1e-9,
                              step), 3)
    acc: list[list[np.ndarray]] = [[] for _ in offs]
    nodata = np.zeros(len(offs))
    total = 0
    t = int_ds.transform
    ps = t.a
    for g in shapely.get_parts(lines):
        L = shapely.length(g)
        if 4 * step > L:
            continue
        b = shapely.bounds(g)
        pad = cfg.profile_half_width_m + 3 * ps
        win = from_bounds(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad, t)
        win = win.round_offsets().round_lengths().intersection(
            Window(0, 0, int_ds.width, int_ds.height))
        if win.width <= 0 or win.height <= 0:
            continue
        im = int_ds.read(1, window=win)
        d = np.arange(step, L - step, step)
        p = shapely.get_coordinates(shapely.line_interpolate_point(g, d))
        pa = shapely.get_coordinates(shapely.line_interpolate_point(g, d - step / 2))
        pb = shapely.get_coordinates(shapely.line_interpolate_point(g, d + step / 2))
        tang = pb - pa
        tang /= np.maximum(np.hypot(*tang.T), 1e-9)[:, None]
        nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
        x0 = t.c + win.col_off * ps
        y0 = t.f - win.row_off * ps
        total += len(p)
        for k, o in enumerate(offs):
            q = p + nrm * o
            col = np.floor((q[:, 0] - x0) / ps).astype(int)
            row = np.floor((y0 - q[:, 1]) / ps).astype(int)
            ok = (col >= 0) & (col < im.shape[1]) & (row >= 0) & (row < im.shape[0])
            v = np.full(len(q), np.nan, np.float32)
            v[ok] = im[row[ok], col[ok]]
            nodata[k] += np.isnan(v).sum()
            acc[k].append(v[np.isfinite(v)])
    if not total:
        return {"samples": 0}
    med, p75 = [], []
    for a in acc:
        v = np.concatenate(a) if a else np.array([])
        med.append(round(float(np.median(v)), 2) if len(v) else None)
        p75.append(round(float(np.quantile(v, 0.75)), 2) if len(v) else None)
    m = np.array([np.nan if v is None else v for v in med])
    if np.isnan(m).all():
        return {"samples_along": int(total)}
    edge = np.nanmedian(np.r_[m[:3], m[-3:]])
    peak = int(np.nanargmax(m))
    half = edge + (m[peak] - edge) / 2
    above = np.flatnonzero(m >= half)
    kc = int(np.argmin(np.abs(offs)))
    ko = [int(np.argmin(np.abs(offs - o))) for o in (-0.45, 0.45)]
    centre = np.concatenate(acc[kc] or [np.array([])])
    away = np.concatenate([a for k in ko for a in acc[k]] or [np.array([])])
    return {"samples_along": int(total),
            # Bright = int_max >= 2 x the profile edge. On the line vs 45 cm beside it:
            # ~1 on solid paint, ~dash/(dash+gap) if a dashed line spans its gaps.
            "bright_share_on_line": round(float((centre >= 2 * edge).mean()), 3)
            if len(centre) else None,
            "bright_share_45cm_beside": round(float((away >= 2 * edge).mean()), 3)
            if len(away) else None,
            "offsets_m": offs.tolist(), "int_max_median": med, "int_max_p75": p75,
            "nodata_share": np.round(nodata / total, 3).tolist(),
            "peak_offset_m": float(offs[peak]),
            "peak_over_edge": round(float(m[peak] / edge), 2) if edge else None,
            "width_at_half_peak_m": round(float((above.max() - above.min() + 1) * step), 3)
            if len(above) else None}


def _profile_by(df, lines: np.ndarray, int_ds, cfg: VectorInventoryCfg) -> dict:
    out: dict = {}
    for f in cfg.group_by:
        if f not in df.columns or not 0 < df[f].nunique() <= cfg.max_values:
            continue
        vals = df[f].astype(str).to_numpy()
        if not all(CODE_LIKE.match(v) for v in set(vals)):
            continue
        out[f] = {}
        for v in sorted(set(vals)):
            p = _profile(lines[vals == v], int_ds, cfg)
            out[f][v] = {k: p[k] for k in p if k not in ("offsets_m", "int_max_p75",
                                                          "nodata_share")}
            if "int_max_median" in p:
                out[f][v]["int_max_median"] = p["int_max_median"][::2]  # every 2 steps
    return out


def _box_contrast(polys: np.ndarray, int_ds) -> dict:
    """Bright-pixel share inside polygons vs a 1 m ring around them."""
    t = int_ds.transform
    inside, ring = [], []
    for g in shapely.get_parts(polys):
        b = shapely.bounds(g)
        win = from_bounds(b[0] - 1.5, b[1] - 1.5, b[2] + 1.5, b[3] + 1.5, t)
        win = win.round_offsets().round_lengths().intersection(
            Window(0, 0, int_ds.width, int_ds.height))
        if win.width <= 0 or win.height <= 0:
            continue
        im = int_ds.read(1, window=win)
        wt = rasterio.windows.transform(win, t)
        shp = (int(win.height), int(win.width))
        inn = features.rasterize([(g, 1)], out_shape=shp, transform=wt).astype(bool)
        out = features.rasterize([(shapely.buffer(g, 1.0), 1)], out_shape=shp,
                                 transform=wt).astype(bool) & ~inn
        inside.append(im[inn])
        ring.append(im[out])
    if not inside:
        return {"polygons": 0}
    a, r = np.concatenate(inside), np.concatenate(ring)
    thr = np.nanquantile(r, 0.9) if np.isfinite(r).any() else np.nan
    return {"polygons": len(inside),
            "int_max_inside": _q(a, 1), "int_max_ring_1m": _q(r, 1),
            "nodata_share_inside": round(float(np.isnan(a).mean()), 3),
            "share_inside_above_ring_p90": round(float(np.nanmean(a > thr)), 3)
            if np.isfinite(thr) else None}


def run_vector_inventory(cfg: VectorInventoryCfg, root: Path) -> tuple[dict, dict, list[Path]]:
    """Returns (redacted report, private name map for the data root, input files)."""
    import geopandas as gpd

    files = _files(root, cfg.paths)
    rects = tile_rects(root, cfg.tiles, cfg.corner_world_file) if cfg.tiles else []
    tile_boxes = shapely.box(*np.array(rects).T) if rects else None
    tree = shapely.STRtree(tile_boxes) if rects else None
    int_ds = dens_ds = None
    grid_box = None
    if cfg.grid:
        gdir = root / cfg.grid
        int_ds = rasterio.open(gdir / "int_max.tif")
        dens_ds = rasterio.open(gdir / "density.tif")
        grid_box = shapely.box(*int_ds.bounds)
    names: dict = {}
    layers_out = []
    try:
        for fi, f in enumerate(files, 1):
            try:
                lyrs = pyogrio.list_layers(f)
            except Exception as e:  # unreadable: report the error class only
                layers_out.append({"file": f"#{fi}", "ext": f.suffix.lower(),
                                   "error": type(e).__name__})
                continue
            names[f"#{fi}"] = {"file": str(f.relative_to(root)), "layers": {}}
            for li, (lname, _gt) in enumerate(lyrs, 1):
                key = f"#{fi}.{li}"
                names[f"#{fi}"]["layers"][key] = lname
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = gpd.read_file(f, layer=lname)
                geoms = df.geometry.to_numpy()
                valid = ~shapely.is_missing(geoms)
                geoms = np.asarray(geoms[valid])
                crs = df.crs
                info: dict = {
                    "layer": key, "ext": f.suffix.lower(), "features": int(len(df)),
                    "empty_geometry": int((~valid).sum()),
                    "geometry_types": dict(Counter(shapely.get_type_id(geoms).tolist())),
                    "has_z": int(shapely.has_z(geoms).sum()),
                    "invalid": int((~shapely.is_valid(geoms)).sum()),
                    "geometry_sha": hashlib.sha256(b"".join(sorted(shapely.to_wkb(
                        shapely.set_precision(geoms, 0.001))))).hexdigest()[:12],
                    "crs": None if crs is None else {
                        "projected": bool(crs.is_projected),
                        "units": crs.axis_info[0].unit_name if crs.axis_info else None},
                    "fields": _fields(df, cfg.max_values),
                }
                info["geometry_types"] = {_TYPES.get(k, str(k)): v
                                          for k, v in info["geometry_types"].items()}
                tid = shapely.get_type_id(geoms)
                lines = geoms[np.isin(tid, (1, 5))]
                polys = geoms[np.isin(tid, (3, 6))]
                if len(lines):
                    info["lines"] = _line_stats(lines)
                if len(polys):
                    info["polygons"] = _polygon_stats(polys)
                if len(geoms):
                    ext = shapely.bounds(geoms)
                    info["layer_extent_m"] = [round(float(ext[:, 2].max() - ext[:, 0].min()),
                                                    1),
                                              round(float(ext[:, 3].max() - ext[:, 1].min()),
                                                    1)]
                if tree is not None and len(geoms):
                    hit = np.zeros(len(geoms), bool)
                    hit[np.unique(tree.query(geoms, predicate="intersects")[0])] = True
                    info["share_on_delivered_tiles"] = round(float(hit.mean()), 4)
                if grid_box is not None and len(geoms):
                    on = shapely.intersects(geoms, grid_box)
                    info["features_on_grid"] = int(on.sum())
                    if on.any():
                        ln = geoms[on & np.isin(tid, (1, 5))]
                        pl = geoms[on & np.isin(tid, (3, 6))]
                        if len(ln):
                            info["line_profile"] = _profile(ln, int_ds, cfg)
                            info["line_profile_by"] = _profile_by(
                                df[valid][on & np.isin(tid, (1, 5))], ln, int_ds, cfg)
                        if len(pl):
                            info["polygon_contrast"] = _box_contrast(pl, int_ds)
                layers_out.append(info)
    finally:
        for ds in (int_ds, dens_ds):
            if ds is not None:
                ds.close()
    rep = {"files": len(files), "by_ext": dict(Counter(f.suffix.lower() for f in files)),
           "delivered_tiles": len(rects), "layers": layers_out}
    return rep, names, files


_TYPES = {0: "Point", 1: "LineString", 2: "LinearRing", 3: "Polygon", 4: "MultiPoint",
          5: "MultiLineString", 6: "MultiPolygon", 7: "GeometryCollection"}
