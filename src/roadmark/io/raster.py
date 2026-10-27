"""Shared raster grid and GeoTIFF/COG writing (one grid, one CRS, one pixel size: §4.3)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.windows import Window


@dataclass(frozen=True)
class Grid:
    """North-up grid.

    Pixel (row, col) covers x in [x0 + col*ps, x0 + (col+1)*ps)
                       and y in (y0 - (row+1)*ps, y0 - row*ps].
    """

    x0: float  # west edge
    y0: float  # north edge
    pixel_size: float
    width: int
    height: int

    @classmethod
    def from_bounds(cls, minx: float, miny: float, maxx: float, maxy: float, ps: float) -> Grid:
        # Snap to multiples of the pixel size so grids from different runs align.
        x0 = math.floor(minx / ps) * ps
        y0 = math.ceil(maxy / ps) * ps
        return cls(x0, y0, ps, int(math.ceil((maxx - x0) / ps)) or 1,
                   int(math.ceil((y0 - miny) / ps)) or 1)

    @property
    def transform(self):
        return from_origin(self.x0, self.y0, self.pixel_size, self.pixel_size)

    def window_bounds(self, w: Window) -> tuple[float, float, float, float]:
        ps = self.pixel_size
        minx = self.x0 + w.col_off * ps
        maxy = self.y0 - w.row_off * ps
        return minx, maxy - w.height * ps, minx + w.width * ps, maxy

    def blocks(self, block_px: int) -> list[Window]:
        return [Window(c, r, min(block_px, self.width - c), min(block_px, self.height - r))
                for r in range(0, self.height, block_px) for c in range(0, self.width, block_px)]

    def cell_index(self, x: np.ndarray, y: np.ndarray, w: Window | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row/col of points relative to window ``w`` (or the full grid), plus in-bounds mask."""
        ps = self.pixel_size
        col = np.floor((x - self.x0) / ps).astype(np.int64)
        row = np.floor((self.y0 - y) / ps).astype(np.int64)
        if w is not None:
            col -= int(w.col_off)
            row -= int(w.row_off)
            wid, hei = int(w.width), int(w.height)
        else:
            wid, hei = self.width, self.height
        ok = (col >= 0) & (col < wid) & (row >= 0) & (row < hei)
        return row, col, ok


def create_tiled(path: Path, grid: Grid, dtype: str, nodata: float | None, crs, description: str,
                 sparse: bool = False) -> None:
    """Create an empty tiled GeoTIFF that blocks are written into with windowed writes.

    ``sparse``: unwritten blocks are not stored and read back as ``nodata`` (0 if None), so
    a mostly empty mosaic costs nothing for the empty part.
    """
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=1, dtype=dtype,
                   crs=crs, transform=grid.transform, nodata=nodata, tiled=True, blockxsize=512,
                   blockysize=512, compress="DEFLATE", BIGTIFF="IF_SAFER")
    if sparse:
        profile["SPARSE_OK"] = "TRUE"
    with rasterio.open(path, "w", **profile) as dst:
        dst.set_band_description(1, description)
        if nodata is not None and not sparse:  # initialise so unwritten areas are not zero
            fill = np.full((512, 512), nodata, dtype=dtype)
            for _, win in dst.block_windows(1):
                dst.write(fill[: win.height, : win.width], 1, window=win)


def write_window(path: Path, arr: np.ndarray, w: Window) -> None:
    with rasterio.open(path, "r+") as dst:
        dst.write(arr, 1, window=w)


def to_cog(src: Path, dst: Path, compress: str = "DEFLATE", resampling: str = "average") -> None:
    """Convert to Cloud-Optimized GeoTIFF with internal overviews (§4.5)."""
    rio_copy(src, dst, driver="COG", compress=compress, overview_resampling=resampling,
             BIGTIFF="IF_SAFER")
