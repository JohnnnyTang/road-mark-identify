import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "check_no_data", Path(__file__).parents[1] / "scripts" / "check_no_data.py")
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _write(root: Path, rel: str, content: bytes | str) -> str:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content.encode() if isinstance(content, str) else content)
    return rel


@pytest.mark.parametrize("rel,content", [
    ("data/anything.txt", "hello"),
    ("runs/x/report.json", "{}"),
    ("configs/local/site.yaml", "crs_epsg: 4547"),
    ("scan.laz", b"LASF...."),
    ("ortho.tif", b"II*\x00...."),
    ("out.gpkg", b"SQLite format 3\x00"),
    ("model.pt", b"PK\x03\x04"),
    ("bundle.zip", b"PK\x03\x04"),
    ("fig.png", b"\x89PNG...."),
    (".env", "ROADMARK_DATA_ROOT=x"),
])
def test_blocks_data(tmp_path, rel, content):
    assert guard.check_file(_write(tmp_path, rel, content), tmp_path)


def test_blocks_renamed_binary(tmp_path):
    rel = _write(tmp_path, "src/innocent.txt", b"LASF" + b"\x00" * 100)
    assert any("signature" in r for r in guard.check_file(rel, tmp_path))


def test_blocks_npy_renamed(tmp_path):
    p = tmp_path / "notes.md"
    np.save(tmp_path / "a.npy", np.zeros(3))
    p.write_bytes((tmp_path / "a.npy").read_bytes())
    assert guard.check_file("notes.md", tmp_path)


@pytest.mark.parametrize("line", [
    "origin = (512345.678, 3456789.012)",  # projected (UTM-like)  data-guard: allow
    "39512345.6 3456789.1",  # Gauss-Krueger with zone prefix  data-guard: allow
    "pt: 31.2345678, 121.4567891",  # geographic  data-guard: allow
])
def test_blocks_coordinates(tmp_path, line):
    rel = _write(tmp_path, "src/x.py", line.split("#")[0] + "\n")
    assert any("coordinates" in r for r in guard.check_file(rel, tmp_path))


def test_pragma_and_clean_code_pass(tmp_path):
    rel = _write(tmp_path, "src/ok.py",
                 "x = 512345.678, 3456789.012  # data-guard: allow\npixel_size = 0.025\n")
    assert guard.check_file(rel, tmp_path) == []


def test_notebook_outputs_blocked(tmp_path):
    nb = '{"cells": [{"cell_type": "code", "outputs": [{"text": "x"}], "source": []}]}'
    assert guard.check_file(_write(tmp_path, "nb/a.ipynb", nb), tmp_path)


def test_size_limit(tmp_path):
    rel = _write(tmp_path, "src/big.txt", "a" * (guard.MAX_BYTES + 1))
    assert any("limit" in r for r in guard.check_file(rel, tmp_path))


def test_public_figures_allowed(tmp_path):
    rel = _write(tmp_path, "docs/figures/public/synthetic.png", b"\x89PNG....")
    assert guard.check_file(rel, tmp_path) == []
