from pathlib import Path

import pytest

from roadmark.synthetic import SceneParams, generate


@pytest.fixture(scope="session")
def synth_root(tmp_path_factory) -> Path:
    """Small synthetic scene in a pytest temp dir (outside the repo)."""
    root = tmp_path_factory.mktemp("data_root")
    generate(root / "synthetic", SceneParams(length=50.0, density=1500.0, seed=1))
    return root
