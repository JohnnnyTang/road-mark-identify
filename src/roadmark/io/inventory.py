"""Delivery inventory (Roadmap 1.1): one streaming pass over many LAS/LAZ tiles.

Per file it builds mergeable histograms (classification, pass ID, intensity, points per
occupied cell) so a delivery of thousands of tiles reduces to one summary per group
(e.g. ground / refined ground / non-ground folders).

Redacted by construction (DATA_POLICY.md): sizes, counts, durations and distributions only.
No bounds, offsets or absolute GPS times are kept, so the report is safe to print.
"""

from __future__ import annotations

import glob
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import laspy
import numpy as np
from pydantic import BaseModel, Field

from roadmark.redact import extent_summary

DENSITY_CAP = 50_000  # points per occupied cell; larger values share the last histogram bin
QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)


class InventoryCfg(BaseModel):
    """Recorded in the manifest. Groups map a name to globs relative to the data root."""

    groups: dict[str, list[str]]
    cell: float = 1.0  # m; occupied-cell density = points per cell / cell²
    chunk_points: int = 5_000_000
    tiny_points: int = 1000  # files below this are listed as near-empty
    workers: int = Field(default=4, exclude=True)


def resolve_groups(cfg: InventoryCfg, root: Path) -> dict[str, list[Path]]:
    out = {}
    for name, pats in cfg.groups.items():
        files: set[Path] = set()
        for pat in pats:
            p = Path(pat) if Path(pat).is_absolute() else root / pat
            files |= {Path(f) for f in glob.glob(str(p))}
        out[name] = sorted(files)
    return out


def file_stats(path: Path, cell: float = 1.0, chunk_points: int = 5_000_000) -> dict:
    """Redacted statistics of one file plus raw histograms for merging (``_hist`` keys)."""
    with laspy.open(path) as f:
        h = f.header
        names = set(h.point_format.dimension_names)
        mins, maxs = h.mins, h.maxs
        ncol = int(np.floor((maxs[0] - mins[0]) / cell)) + 1
        nrow = int(np.floor((maxs[1] - mins[1]) / cell)) + 1
        counts = np.zeros(nrow * ncol, np.int64)
        cls: Counter = Counter()
        passes: dict[int, list[float]] = {}  # id -> [n, t_min, t_max]
        inten = np.zeros(65536, np.int64)
        try:
            crs = h.parse_crs()
            epsg = crs.to_epsg() if crs is not None else None
        except Exception:
            epsg = None
        for pts in f.chunk_iterator(chunk_points):
            col = np.floor((np.asarray(pts.x) - mins[0]) / cell).astype(np.int64)
            row = np.floor((np.asarray(pts.y) - mins[1]) / cell).astype(np.int64)
            counts += np.bincount(np.clip(row, 0, nrow - 1) * ncol + np.clip(col, 0, ncol - 1),
                                  minlength=nrow * ncol)
            cls.update(dict(zip(*np.unique(np.asarray(pts.classification), return_counts=True),
                                strict=True)))
            inten += np.bincount(np.asarray(pts.intensity, np.int64), minlength=65536)
            if "point_source_id" in names:
                pid = np.asarray(pts.point_source_id)
                t = np.asarray(pts.gps_time) if "gps_time" in names else None
                for p in np.unique(pid):
                    sel = pid == p
                    n = int(sel.sum())
                    t0, t1 = (float(t[sel].min()), float(t[sel].max())) if t is not None \
                        else (np.nan, np.nan)
                    a = passes.setdefault(int(p), [0, np.inf, -np.inf])
                    a[0] += n
                    a[1], a[2] = min(a[1], t0), max(a[2], t1)
        occ = counts[counts > 0]
        return {
            "file": path.name,
            "points": int(h.point_count),
            "point_format": h.point_format.id,
            "scale_m": float(h.scales[0]),
            "epsg": epsg,
            "gps_time_type": h.global_encoding.gps_time_type.name.lower(),
            "fields": sorted(names),
            "extent": extent_summary((mins[0], mins[1], maxs[0], maxs[1], mins[2], maxs[2])),
            "occupied_m2": round(len(occ) * cell * cell, 1),
            "_cls_hist": {int(k): int(v) for k, v in cls.items()},
            "_pass_hist": {k: v for k, v in passes.items()},
            "_int_hist": inten,
            "_dens_hist": np.bincount(np.minimum(occ, DENSITY_CAP), minlength=DENSITY_CAP + 1),
        }


def hist_quantiles(hist: np.ndarray, qs=QUANTILES, scale: float = 1.0) -> dict[str, float]:
    """Quantiles of a value histogram (bin i = value i), keyed 'p50' etc."""
    total = hist.sum()
    if total == 0:
        return {}
    cdf = np.cumsum(hist) / total
    # Smallest value with CDF >= q; q = 0 means the smallest observed value.
    return {f"p{q * 100:g}": float(np.searchsorted(cdf, max(q, 1e-12)) * scale) for q in qs}


class GroupSummary:
    """Merges per-file stats of one group as they arrive (histograms are ~1 MB per file)."""

    def __init__(self, cell: float, tiny_points: int):
        self.cell, self.tiny_points = cell, tiny_points
        self.rows: list[dict] = []
        self.cls: Counter = Counter()
        self.passes: dict[int, list[float]] = {}
        self.inten = np.zeros(65536, np.int64)
        self.dens = np.zeros(DENSITY_CAP + 1, np.int64)

    def add(self, s: dict) -> None:
        self.cls.update(s["_cls_hist"])
        self.inten += s["_int_hist"]
        self.dens += s["_dens_hist"]
        for p, (n, t0, t1) in s["_pass_hist"].items():
            a = self.passes.setdefault(p, [0, np.inf, -np.inf])
            a[0] += n
            a[1], a[2] = min(a[1], t0), max(a[2], t1)
        self.rows.append({k: v for k, v in s.items() if not k.startswith("_")})

    def summary(self) -> dict:
        rows = self.rows
        if not rows:
            return {"files": 0}
        npts = sum(r["points"] for r in rows)
        occ = sum(r["occupied_m2"] for r in rows)
        spans = [t1 - t0 for _, t0, t1 in self.passes.values() if np.isfinite(t1 - t0)]
        common = set.intersection(*(set(r["fields"]) for r in rows))
        some = set.union(*(set(r["fields"]) for r in rows)) - common
        per_pass = np.array([v[0] for v in self.passes.values()], np.int64)
        return {
            "files": len(rows),
            "points": npts,
            "near_empty_files": sum(r["points"] < self.tiny_points for r in rows),
            "point_formats": dict(Counter(r["point_format"] for r in rows)),
            "scale_m": sorted({r["scale_m"] for r in rows}),
            "epsg": sorted({str(r["epsg"]) for r in rows}),
            "gps_time_type": dict(Counter(r["gps_time_type"] for r in rows)),
            "fields_in_all": sorted(common),
            "fields_in_some": sorted(some),
            "occupied_area_m2": round(occ, 1),
            "density_on_occupied_pts_m2": {
                "mean": round(npts / max(occ, 1e-9), 1),
                **hist_quantiles(self.dens, (0.05, 0.25, 0.5, 0.75, 0.95),
                                 1.0 / (self.cell * self.cell)),
            },
            "classification": {str(k): v for k, v in sorted(self.cls.items())},
            "passes": {
                "count": len(self.passes),
                "points_per_pass": {"min": int(per_pass.min()), "median": int(np.median(per_pass)),
                                    "max": int(per_pass.max())} if len(per_pass) else None,
                "span_s": {"min": round(min(spans), 1), "median": round(float(np.median(spans)), 1),
                           "max": round(max(spans), 1)} if spans else None,
            },
            "intensity": {"max": int(np.flatnonzero(self.inten).max()) if self.inten.any()
                          else None, **hist_quantiles(self.inten)},
        }


def _stats_job(args: tuple[Path, float, int]) -> dict:
    return file_stats(*args)


def run_inventory(cfg: InventoryCfg, root: Path) -> tuple[dict, list[Path]]:
    """Returns (report, all input files). The report holds no coordinates."""
    groups = resolve_groups(cfg, root)
    report: dict = {"cell_m": cfg.cell, "groups": {}, "files": {}}
    all_files: list[Path] = []
    with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
        for name, files in groups.items():
            acc = GroupSummary(cfg.cell, cfg.tiny_points)
            for s in ex.map(_stats_job, [(f, cfg.cell, cfg.chunk_points) for f in files],
                            chunksize=4):
                acc.add(s)
            report["groups"][name] = acc.summary()
            report["files"][name] = acc.rows
            all_files += files
    report["tiles"] = tile_sets({n: [f.name for f in fs] for n, fs in groups.items()})
    return report, all_files


def tile_key(name: str) -> str:
    """Tile identity shared across groups: 'grid_1_2_ng.las' -> 'grid_1_2'."""
    stem = name.split(".", 1)[0]
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def tile_sets(groups: dict[str, list[str]]) -> dict:
    """How the groups' tile sets overlap (counts only)."""
    keys = {n: {tile_key(f) for f in fs} for n, fs in groups.items()}
    out = {}
    names = list(keys)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}|{b}"] = {"both": len(keys[a] & keys[b]), f"only_{a}": len(keys[a] - keys[b]),
                               f"only_{b}": len(keys[b] - keys[a])}
    return out
