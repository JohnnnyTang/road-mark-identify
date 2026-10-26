"""Data-root layout, shared by the CPU (Stage 1, 3-7) and GPU (Stage 2, glyph CNN) machines."""

from __future__ import annotations

from pathlib import Path

from roadmark.config import default_data_root

LAYOUT = {
    "raw": "delivered point clouds and trajectories (read-only, never modified)",
    "synthetic": "output of `roadmark synth`",
    "annotations": "vector annotations in the project CRS: training paint, ignore areas, "
                   "gold standard (§8.2 schema), split areas",
    "runs": "pipeline outputs per run: runs/<name>/stage1, stage2, ... each with manifest.json",
    "models": "trained weights (GPU machine), with their training manifest",
    "exchange": "bundles copied between the CPU and GPU machines (see DATA_POLICY.md)",
}

README = """# Data root (never commit, never upload)

Created by `roadmark init`. Everything here is project data: see DATA_POLICY.md in the repo.

{items}
"""


def init_data_root(root: Path | None = None) -> Path:
    root = root or default_data_root()
    for name in LAYOUT:
        (root / name).mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        items = "\n".join(f"- `{k}/`: {v}" for k, v in LAYOUT.items())
        readme.write_text(README.format(items=items), encoding="utf-8")
    return root
