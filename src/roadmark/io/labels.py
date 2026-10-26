"""Vector annotations -> label rasters on any grid.

Annotations are drawn as **vectors in the project CRS**, never as pixels. They don't depend
on the raster grid, so annotation can start on preview rasters (Roadmap milestone M1) and
survive later changes of pixel size or intensity correction (M2).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from rasterio import features
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry.base import BaseGeometry

from roadmark.io.raster import Grid

IGNORE = 255  # excluded from loss and metrics (§5.1)


def rasterize_polygons(shapes: Iterable[tuple[BaseGeometry, int]], grid: Grid,
                       window: Window | None = None, fill: int = 0,
                       all_touched: bool = False) -> np.ndarray:
    """Burn (geometry, value) pairs into a uint8 array covering ``window`` (or the grid)."""
    w = window or Window(0, 0, grid.width, grid.height)
    shapes = list(shapes)
    out_shape = (int(w.height), int(w.width))
    if not shapes:
        return np.full(out_shape, fill, dtype=np.uint8)
    return features.rasterize(shapes, out_shape=out_shape, fill=fill, dtype="uint8",
                              transform=window_transform(w, grid.transform),
                              all_touched=all_touched)


def paint_labels(paint: Iterable[BaseGeometry], grid: Grid, coverage: np.ndarray,
                 ignore: Iterable[BaseGeometry] = (), window: Window | None = None
                 ) -> np.ndarray:
    """1 = paint, 0 = observed background, 255 = ignore (no data or annotator-ignored)."""
    lab = rasterize_polygons(((g, 1) for g in paint), grid, window)
    ign = rasterize_polygons(((g, 1) for g in ignore), grid, window)
    lab[(coverage == 0) | (ign == 1)] = IGNORE
    return lab
