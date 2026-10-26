from pathlib import Path

import pytest

from roadmark.synthetic import SceneParams, generate


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    """Small synthetic scene in a pytest temp dir (outside the repo)."""
    root = tmp_path_factory.mktemp("data_root")
    generate(root / "synthetic", SceneParams(length=50.0, density=1500.0, seed=1))
    return root


@pytest.fixture(scope="session")
def stage1_run(synth_root) -> tuple[dict, Path]:
    """Stage 1 on the synthetic scene at 5 cm (fast), several blocks."""
    from roadmark.config import load_pipeline
    from roadmark.stage1_raster.pipeline import run_stage1

    mp = pytest.MonkeyPatch()
    mp.setenv("ROADMARK_DATA_ROOT", str(synth_root))
    try:
        cfg = load_pipeline("configs/pipeline/default.yaml", "configs/pipeline/synthetic.yaml")
        cfg.stage1.raster.pixel_size = 0.05
        cfg.stage1.raster.block_size_px = 300
        rep = run_stage1(cfg)
    finally:
        mp.undo()
    return rep, synth_root / "runs" / "synthetic" / "stage1"
