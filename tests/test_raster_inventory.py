import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from roadmark.io.raster_inventory import RasterInventoryCfg, run_raster_inventory, tile_name_index


def _tile(path, ix, iy, value, ps=0.5, n=20, nodata=None, world_file=False):
    """Tile (ix, iy) of a local grid: west edge ix*n*ps, south edge iy*n*ps."""
    arr = np.full((1, n, n), value, np.float32)
    arr[0, :2, :] = np.nan if nodata is None else nodata
    t = from_origin(ix * n * ps, (iy + 1) * n * ps, ps, ps)
    prof = dict(driver="GTiff", width=n, height=n, count=1, dtype="float32", nodata=nodata)
    if world_file:  # georeferenced only by a .tfw sidecar, no CRS (like the delivery)
        with rasterio.open(path, "w", **prof) as ds:
            ds.write(arr)
        path.with_suffix(".tfw").write_text(
            "\n".join(str(v) for v in (ps, 0, 0, -ps, t.c + ps / 2, t.f - ps / 2)))
    else:
        with rasterio.open(path, "w", transform=t, **prof) as ds:
            ds.write(arr)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_raster_inventory(tmp_path):
    for ch, v in (("density", 5.0), ("intensity", 0.0)):
        (tmp_path / ch).mkdir()
        for ix, iy in ((-1, 3), (0, 3), (0, 4)):
            _tile(tmp_path / ch / f"{ix}_{iy}.tif", ix, iy, v, world_file=(ch == "density"),
                  nodata=-9999.0 if ch == "intensity" else None)
    cfg = RasterInventoryCfg(groups={"density": ["density/*.tif"],
                                     "intensity": ["intensity/*.tif"]})
    rep, files = run_raster_inventory(cfg, tmp_path)
    d, i = rep["groups"]["density"], rep["groups"]["intensity"]
    assert len(files) == 6 and d["files"] == 3
    assert d["pixel_size_m"] == {"(0.5, 0.5)": 3} and d["crs"] == {"None": 3}
    assert d["grid"]["tile_size_m"] == [10.0, 10.0] and d["grid"]["origins_on_pixel_grid"] == 3
    assert d["grid"]["name_matches_position"] == {"(floor(minx/W), floor(miny/H))": 3}
    assert d["bands"][0]["nodata_frac"] == 0.1 and d["bands"][0]["p50"] == 5.0
    assert i["nodata_value"] == {"-9999.0": 3} and i["bands"][0]["zero_frac_of_valid"] == 1.0
    assert rep["cross_group"]["density|intensity"] == {"common_tiles": 3, "identical_grid": 3}
    assert "_origin" not in json.dumps(rep)  # positions never reach the report


def test_tile_name_index():
    assert tile_name_index("-2_5.tif") == (-2, 5)
    assert tile_name_index("grid_-3_7_g.las") == (-3, 7)
    assert tile_name_index("intensity.zip") is None
