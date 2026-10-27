import json
import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from roadmark.io.pointcloud import iter_chunks
from roadmark.stage1_raster.raster_adapter import AdapterCfg, run_adapter
from roadmark.stage1_raster.rasterize import CHANNELS

PS, TILE, PITCH = 0.1, 120, 100  # 12 m tiles every 10 m: 2 m overlaps, like the delivery's 24 m
DROP = "1_1"  # a missing tile: its neighbours' overlaps still cover part of it


@pytest.fixture(scope="module")
def delivery(synth_root):
    """Truth rasters from the synthetic ground points, encoded like the delivered tiles.

    Every tile is wrong where another tile's centre is nearer (intensity + 7): the adapter must
    take each pixel from its nearest-centre tile, so the output is still exact.
    """
    pts = next(iter_chunks(synth_root / "synthetic" / "pass_0.laz", 10**8))
    g = pts["classification"] == 2
    x, y = pts["x"][g], pts["y"][g]
    it = np.minimum(pts["intensity"][g] // 256, 60).astype(np.int64)  # some saturate at 45
    x0, ytop = math.floor(x.min()) - 1.0, math.ceil(y.max()) + 1.0
    nx = math.ceil((x.max() - x0) / (PITCH * PS))
    ny = math.ceil((ytop - y.min()) / (PITCH * PS))
    nr, nc = (ny - 1) * PITCH + TILE, (nx - 1) * PITCH + TILE
    col = np.floor((x - x0) / PS).astype(np.int64)
    row = np.floor((ytop - y) / PS).astype(np.int64)
    cnt = np.zeros((nr, nc), np.int64)
    np.add.at(cnt, (row, col), 1)
    imax = np.zeros((nr, nc), np.int64)
    np.maximum.at(imax, (row, col), it)

    tiles = sorted((f"{i}_{j}", i, j) for i in range(nx) for j in range(ny) if f"{i}_{j}" != DROP)
    rr, cc = np.mgrid[0:nr, 0:nc] + 0.5
    d2 = np.full((len(tiles), nr, nc), np.inf)
    for k, (_, i, j) in enumerate(tiles):
        r0, c0 = j * PITCH, i * PITCH
        sl = np.s_[r0:r0 + TILE, c0:c0 + TILE]
        d2[k][sl] = (rr[sl] - r0 - TILE / 2) ** 2 + (cc[sl] - c0 - TILE / 2) ** 2
    owner = np.where(np.isfinite(d2).any(0), d2.argmin(0), -1)

    dens_v = np.floor(5.1 * np.minimum(cnt, 50)).astype(np.uint8)
    int_v = np.where(cnt > 0, np.floor(255 * np.minimum(imax, 45) / 45), 0).astype(np.uint8)
    for ch in ("density", "intensity", "height"):
        (synth_root / "deliv" / ch).mkdir(parents=True)
    for k, (stem, i, j) in enumerate(tiles):
        r0, c0 = j * PITCH, i * PITCH
        sl = np.s_[r0:r0 + TILE, c0:c0 + TILE]
        bad = (owner[sl] != k) & (cnt[sl] > 0)
        arrs = {"density": dens_v[sl],
                "intensity": np.where(bad, np.minimum(int_v[sl] + 7, 255), int_v[sl]),
                "height": np.zeros((TILE, TILE))}
        tx, ty = x0 + c0 * PS, ytop - r0 * PS
        for ch, a in arrs.items():
            # The world file holds the pixel corner, so the GeoTIFF origin is off by half a pixel.
            with rasterio.open(synth_root / "deliv" / ch / f"{stem}.tif", "w", driver="GTiff",
                               width=TILE, height=TILE, count=1, dtype="uint8",
                               transform=from_origin(tx - PS / 2, ty + PS / 2, PS, PS)) as ds:
                ds.write(a.astype(np.uint8), 1)
    covered = owner >= 0
    return {"x0": x0, "ytop": ytop, "count": np.where(covered, np.minimum(cnt, 50), 0),
            "int_max": np.where(covered & (cnt > 0), np.minimum(imax, 45), np.nan),
            "n_tiles": len(tiles), "covered": covered}


def _cfg(**kw):
    return AdapterCfg(channels={"density": ["deliv/density/*.tif"],
                                "intensity": ["deliv/intensity/*.tif"]}, block_px=256, **kw)


def _read(path):
    with rasterio.open(path) as ds:
        return ds.read(1), ds.transform


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_adapter_decodes_and_mosaics_exactly(synth_root, delivery):
    out = synth_root / "runs" / "adapt" / "stage1"
    rep, inputs = run_adapter(_cfg(), synth_root, out)
    assert rep["tiles"] == delivery["n_tiles"] and len(inputs) == 2 * delivery["n_tiles"]
    dens, t = _read(out / "density.tif")
    # Half-pixel corner fix: the output grid starts at the true pixel corner.
    assert t.c == pytest.approx(delivery["x0"]) and t.f == pytest.approx(delivery["ytop"])
    assert t.a == pytest.approx(PS)
    h, w = delivery["count"].shape
    assert dens.shape == (h, w)
    np.testing.assert_array_equal(dens, delivery["count"])
    imax, _ = _read(out / "int_max.tif")
    np.testing.assert_array_equal(imax, delivery["int_max"])  # NaN exactly where no points
    cov, _ = _read(out / "coverage.tif")
    np.testing.assert_array_equal(cov, (delivery["count"] >= 1).astype(np.uint8))
    for k in ("int_mean", "int_p90", "hag_max", "ground_z"):
        a, _ = _read(out / f"{k}.tif")
        assert np.isnan(a).all()  # not recoverable from the delivery: no data, never zero
    assert sorted(rep["outputs"]) == sorted(f"{k}.tif" for k in CHANNELS)
    ov = rep["overlaps"]
    assert ov["pairs"] > 0
    assert ov["all"]["density_equal_of_either"] == 1.0
    assert ov["all"]["intensity_equal_of_both"] < 1.0  # the corrupted halves are seen
    assert ov["all"]["owner_empty_other_occupied_of_either"] == 0.0
    c = rep["coverage"]
    assert c["occupied_px"] == int((delivery["count"] > 0).sum())
    assert 0 < c["intensity_saturated_frac_of_occupied"] < 1
    blob = json.dumps(rep)
    assert "x0" not in blob and "transform" not in blob  # redacted: no positions


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_adapter_aggregates_2x2(synth_root, delivery):
    out = synth_root / "runs" / "adapt2" / "stage1"
    rep, _ = run_adapter(_cfg(aggregate=2, coverage_min_points=2), synth_root, out)
    cnt, imax = delivery["count"], delivery["int_max"]
    # Tile corners are whole metres: the native grid already starts on the 2-pixel lattice.
    h, w = (s // 2 * 2 for s in cnt.shape)
    want_c = cnt[:h, :w].reshape(h // 2, 2, w // 2, 2).sum(axis=(1, 3))
    want_i = np.fmax.reduce(np.fmax.reduce(imax[:h, :w].reshape(h // 2, 2, w // 2, 2), 3), 1)
    dens, t = _read(out / "density.tif")
    assert t.a == pytest.approx(2 * PS) and rep["grid"]["pixel_size"] == pytest.approx(2 * PS)
    np.testing.assert_array_equal(dens, want_c)
    np.testing.assert_array_equal(_read(out / "int_max.tif")[0], want_i)
    np.testing.assert_array_equal(_read(out / "coverage.tif")[0], want_c >= 2)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_adapter_guards(synth_root, delivery):
    out = synth_root / "runs" / "adapt_bad" / "stage1"
    with pytest.raises(ValueError, match="pixel lattice"):  # no corner fix: half-pixel origins
        run_adapter(_cfg(corner_world_file=False), synth_root, out)
    with pytest.raises(ValueError, match="max_grid_mpx"):
        run_adapter(_cfg(max_grid_mpx=0.001), synth_root, out)
    with pytest.raises(ValueError, match="height"):
        run_adapter(AdapterCfg(channels={"density": ["deliv/density/*.tif"],
                                         "height": ["deliv/height/*.tif"]}), synth_root, out)


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_adapt_rasters_cli(synth_root, delivery, monkeypatch):
    from typer.testing import CliRunner

    from roadmark.cli import app

    monkeypatch.setenv("ROADMARK_DATA_ROOT", str(synth_root))
    res = CliRunner().invoke(app, ["adapt-rasters", "density=deliv/density/*.tif",
                                   "intensity=deliv/intensity/*.tif", "--name", "adapt_cli"])
    assert res.exit_code == 0, res.output
    out = synth_root / "runs" / "adapt_cli" / "stage1"
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["stage"] == "adapt_rasters" and "unavailable" in man["channels"]["int_mean"]
    assert json.loads(res.output)["tiles"] == delivery["n_tiles"]
    assert (out / "qa_adapter.json").is_file() and not (out / "work").exists()


@pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")
def test_adapt_rasters_cli_pick(synth_root, delivery, monkeypatch):
    from typer.testing import CliRunner

    from roadmark.cli import app

    monkeypatch.setenv("ROADMARK_DATA_ROOT", str(synth_root))
    res = CliRunner().invoke(app, ["adapt-rasters", "density=deliv/density/*.tif",
                                   "intensity=deliv/intensity/*.tif", "--name", "adapt_pick",
                                   "--pick", "2", "--workers", "1"])
    assert res.exit_code == 0, res.output
    rep = json.loads(res.output)
    assert rep["tiles"] == 4 and rep["pilot_pick"]["picked_tiles"] == 4
    assert "tiles" not in rep["pilot_pick"]  # names stay in the data root
    picked = json.loads((synth_root / "runs" / "adapt_pick" / "pilot_tiles.json").read_text())
    assert DROP not in picked["tiles"]  # only complete 2 x 2 groups qualify
    assert rep["grid"]["width_px"] == PITCH + TILE and rep["overlaps"]["pairs"] == 6


def test_overlap_stats_skip_empty_pairs():
    from roadmark.stage1_raster.raster_adapter import _OverlapStats

    s = _OverlapStats()
    z = np.zeros((2, 2), np.uint16)
    s.add(z, z, z, z, np.ones((2, 2), bool))  # nothing observed: no evidence either way
    d = np.array([[1, 0], [2, 0]], np.uint16)
    s.add(d, d, d, d + 1, np.ones((2, 2), bool))
    rep = s.report()
    assert rep["pairs"] == 2 and rep["pairs_with_points"] == 1
    assert rep["worst_pair"]["density_equal_of_either"] == 1.0
    assert rep["worst_pair"]["intensity_equal_of_both"] == 0.0
