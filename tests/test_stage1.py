import json

import numpy as np
import pytest
import rasterio

from roadmark.config import IntensityCfg, PipelineConfig, load_pipeline, load_standard
from roadmark.io.pointcloud import iter_chunks, read_header
from roadmark.io.raster import Grid
from roadmark.stage1_raster import intensity as I
from roadmark.stage1_raster.pipeline import run_stage1
from roadmark.stage1_raster.rasterize import cell_stats
from roadmark.synthetic import CAR


def test_configs_load():
    cfg = load_pipeline("configs/pipeline/default.yaml", "configs/pipeline/synthetic.yaml")
    assert cfg.allow_local_crs and cfg.stage1.raster.pixel_size == 0.025
    std = load_standard(cfg.standard)
    assert std.verified is False  # must stay visible until someone checks the standard


def test_crs_required():
    with pytest.raises(ValueError, match="crs_epsg"):
        PipelineConfig()


def test_grid_snapping_and_index():
    g = Grid.from_bounds(0.01, 0.01, 1.0, 0.5, 0.025)
    assert g.x0 == 0.0 and g.y0 == 0.5 and g.width == 40 and g.height == 20
    r, c, ok = g.cell_index(np.array([0.0124, 0.999]), np.array([0.49, 0.011]))
    assert ok.all() and (r[0], c[0]) == (0, 0) and (r[1], c[1]) == (19, 39)


def test_cell_stats_exact():
    row = np.array([0, 0, 0, 1])
    col = np.array([0, 0, 0, 1])
    inten = np.array([1.0, 3.0, 2.0, 5.0])
    ch = cell_stats(row, col, (2, 2), inten, np.zeros(4))
    assert ch["int_max"][0, 0] == 3 and ch["int_mean"][0, 0] == 2 and ch["int_p90"][0, 0] == 3
    assert ch["density"][0, 0] == 3 and ch["density"][1, 1] == 1
    assert np.isnan(ch["int_max"][0, 1])  # no points -> no data, not zero


def test_intensity_model_recovers_truth(synth_root):
    hdr = read_header(synth_root / "synthetic" / "pass_0.laz")
    assert hdr.point_count > 0 and "gps_time" in hdr.fields
    pts = [next(iter_chunks(synth_root / "synthetic" / f"pass_{i}.laz", 10**8)) for i in (0, 1)]
    pts = {k: np.concatenate([p[k] for p in pts]) for k in pts[0]}
    # Contract: the fit receives in-road ground points (the pipeline uses the road mask).
    g = (pts["classification"] == 2) & (np.abs(pts["y"]) < 6.0)
    d = np.load(synth_root / "synthetic" / "trajectory.npz")
    traj = {k: d[k] for k in ("t", "x", "y", "z")}
    r, c = I.scanner_geometry(pts, traj, 0.05)
    pid = pts["point_source_id"][g].astype(int)
    m = I.fit_model(pts["intensity"][g], r[g], c[g], pid, IntensityCfg(), seed=0)
    # Flat road + profile scanner: cos = h/R, so only a + b is identifiable (true 1.2 + 0.8).
    assert not m.identifiable
    assert m.a + m.b == pytest.approx(2.0, abs=0.1)
    g1, g2 = m.pass_affine[1][0], m.pass_affine[2][0]
    assert g2 / g1 == pytest.approx(1 / 0.7, rel=0.1)  # pass 2 had 0.7x gain
    # After correction, asphalt no longer depends on range.
    corr = m.correct(pts["intensity"][g], r[g], c[g], pid)
    asphalt = (pts["y"][g] > -5.4) & (pts["y"][g] < -2.2)  # inside the right lane, no paint
    lr = np.log(r[g][asphalt])
    assert abs(np.corrcoef(lr, np.log(corr[asphalt]))[0, 1]) < 0.05
    assert m.asphalt_fraction > 0.5


def test_stage1_end_to_end(synth_root, monkeypatch):
    monkeypatch.setenv("ROADMARK_DATA_ROOT", str(synth_root))
    cfg = load_pipeline("configs/pipeline/default.yaml", "configs/pipeline/synthetic.yaml")
    cfg.stage1.raster.pixel_size = 0.05  # keep the test fast
    cfg.stage1.raster.block_size_px = 300
    rep = run_stage1(cfg)
    out = synth_root / "runs" / "synthetic" / "stage1"
    assert (out / "manifest.json").exists() and (out / "qa_stage1.json").exists()
    assert rep["grid"]["blocks"] > 1
    assert rep["cross_pass_consistency"]["passed"], rep["cross_pass_consistency"]
    m = rep["intensity_model"]
    assert m["a"] + m["b"] == pytest.approx(2.0, abs=0.1)

    with rasterio.open(out / "coverage.tif") as src:
        cov = src.read(1)
        assert src.profile["tiled"]
        row, col = src.index((CAR["x0"] + CAR["x1"]) / 2, (CAR["y0"] + CAR["y1"]) / 2)
        assert cov[row, col] == 0  # under the car: no data, not "no paint"
        r2, c2 = src.index(10.0, -1.0)
        assert cov[r2, c2] == 1
    with rasterio.open(out / "int_mean.tif") as src:
        im = src.read(1)
        assert np.isnan(im[row, col])
        rp, cp = src.index(10.0, -5.6)  # solid edge line
        ra, ca = src.index(10.0, -4.0)  # asphalt
        assert im[rp, cp] > 3 * im[ra, ca]
    with rasterio.open(out / "road_mask.tif") as src:
        road = src.read(1)
        assert road[src.index(10.0, -3.0)] == 1 and road[src.index(10.0, -7.0)] == 0

    report = json.loads((out / "qa_stage1.json").read_text())
    assert "bounds" not in json.dumps(report["inputs"])  # redacted by default
