# CLAUDE.md

Road-marking extraction from point clouds. Design: `docs/handoff.md`. Plan: `docs/ROADMAP.md`,
organised as parallel tracks (A data, B specs, C annotation, D downstream, E model, F classical)
that meet at milestones M1–M5 through the frozen interface contracts I1–I8. Work inside one
track at a time. Don't change an interface without updating its contract, both sides and their tests.
A unit of work closes only when its acceptance check passes and `docs/DECISIONS.md` has an entry.

Two machines: this CPU workstation, and an NVIDIA GPU machine for model code (`seg-cuda` extra,
`device.get_device(require_cuda=True)`). Model code must also run on CPU for synthetic smoke tests.
Data root: `<repo>/data` unless `ROADMARK_DATA_ROOT` is set.

## Data protection (hard rules)

This repo is **public**; the project data is not. See `DATA_POLICY.md`.

- Never open, print, cat, head or sample real data files: `*.las/laz/e57/ply`, rasters
  (`*.tif`), vectors (`*.gpkg/shp/geojson`), arrays (`*.npy/npz`), anything under the data root
  (`$ROADMARK_DATA_ROOT`, `data/`, `runs/`, `models/`) or `configs/local/`. Work through the
  CLI's redacted summaries (`roadmark inspect`, `qa_stage1.json` fields the user shares).
  Synthetic data that you generated yourself into a scratch directory is fine.
- Never send data, coordinates, extents, place names, screenshots or derived statistics to any
  web service, artifact, document, gist, issue, PR text, or search query.
- Never write absolute coordinates into code, configs, tests or docs; use the synthetic
  scene's local coordinates. If a test really needs a coordinate-looking literal, mark the line
  `# data-guard: allow` and explain why.
- Never `git add -f`, never `--no-verify`, never edit `.gitignore` or `scripts/check_no_data.py`
  to let a file through. If the guard blocks something, stop and ask.
- Ask before every `git push` and before creating issues, PRs or releases.
- Do not upload models or experiment logs anywhere (no HF Hub, W&B cloud, cloud buckets).

## Conventions

- Python 3.12, uv. `uv run pytest`, `uv run ruff check .` before committing.
- Develop and test every stage on `roadmark.synthetic` first; tests use pytest temp dirs.
- Metric projected CRS only after rasterization; all thresholds in metres; dimensional
  parameters come from `configs/standards/*.yaml`, never hard-coded.
- Three-state evidence everywhere: paint / no paint / no data. No-data is NaN (floats) or
  coverage = 0, never zero intensity, never "background".
- Precision first: abstain (`unknown`, review flag) rather than guess; thresholds are fitted on
  validation data, not tuned by hand.
- Outputs are written only under `<data_root>/runs/<name>/` with a `manifest.json`.
- CLI output is redacted by default (sizes, counts); absolute coordinates only with
  `--show-coords`, for local use.
