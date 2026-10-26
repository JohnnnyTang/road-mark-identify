"""LAS/LAZ reading: header inspection and chunked iteration into plain NumPy arrays."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import laspy
import numpy as np

from roadmark.redact import extent_summary

# Fields carried through Stage 1 when present in the point format.
OPTIONAL_FIELDS = ("gps_time", "point_source_id", "scan_angle_rank", "scan_angle", "user_data",
                   "red", "green", "blue")


@dataclass
class Header:
    path: Path
    point_count: int
    point_format: int
    bounds: tuple[float, float, float, float, float, float]  # minx, miny, maxx, maxy, minz, maxz
    fields: list[str]
    epsg: int | None
    crs_is_projected: bool | None

    def summary(self, show_coords: bool = False) -> dict:
        b = self.bounds
        ext = extent_summary(b, show_coords)
        return {
            "file": self.path.name,
            "points": self.point_count,
            "point_format": self.point_format,
            "density_pts_m2": round(self.point_count / max(ext["area_m2"], 1e-9), 1),
            "extent": ext,
            "epsg": self.epsg,
            "crs_projected": self.crs_is_projected,
            "has": {f: f in self.fields for f in ("intensity", "classification", *OPTIONAL_FIELDS)},
        }


def read_header(path: str | Path) -> Header:
    path = Path(path)
    with laspy.open(path) as f:
        h = f.header
        epsg, projected = None, None
        try:
            crs = h.parse_crs()
            if crs is not None:
                epsg, projected = crs.to_epsg(), bool(crs.is_projected)
        except Exception:  # malformed VLRs: report unknown, don't fail inspection
            pass
        fields = list(h.point_format.dimension_names)
        return Header(path=path, point_count=int(h.point_count), point_format=h.point_format.id,
                      bounds=(*h.mins[:2], *h.maxs[:2], h.mins[2], h.maxs[2]),
                      fields=fields, epsg=epsg, crs_is_projected=projected)


def _to_arrays(pts: laspy.ScaleAwarePointRecord, names: list[str]) -> dict[str, np.ndarray]:
    out = {"x": np.asarray(pts.x, dtype=np.float64), "y": np.asarray(pts.y, dtype=np.float64),
           "z": np.asarray(pts.z, dtype=np.float64),
           "intensity": np.asarray(pts.intensity, dtype=np.float32),
           "classification": np.asarray(pts.classification, dtype=np.uint8)}
    for f in OPTIONAL_FIELDS:
        if f in names:
            out[f] = np.asarray(pts[f])
    return out


def iter_chunks(path: str | Path, chunk_points: int = 5_000_000) -> Iterator[dict[str, np.ndarray]]:
    with laspy.open(Path(path)) as f:
        names = list(f.header.point_format.dimension_names)
        for pts in f.chunk_iterator(chunk_points):
            yield _to_arrays(pts, names)


def write_las(path: str | Path, arrays: dict[str, np.ndarray], epsg: int | None = None,
              point_format: int = 1, scale: float = 0.001) -> None:
    """Write arrays to LAS/LAZ (used for synthetic data and debugging exports)."""
    header = laspy.LasHeader(point_format=point_format, version="1.4")
    header.scales = np.array([scale] * 3)
    header.offsets = np.array([np.floor(arrays[k].min()) for k in ("x", "y", "z")])
    if epsg is not None:
        from pyproj import CRS
        header.add_crs(CRS.from_epsg(epsg))
    las = laspy.LasData(header)
    for k, v in arrays.items():
        setattr(las, k, v)
    las.write(str(path))
