# roadmark

Extraction of road surface markings from mobile/airborne laser-scanning point clouds, delivered
as complete, typed vector objects (lane lines with solid/dashed intervals, stop lines,
crosswalks, arrows, hatched areas…) with confidence, evidence status and provenance.

**Precision first:** when evidence is insufficient the pipeline says `unknown` or flags the
object for review. Missing data is never treated as absence of paint.

> **Data policy:** this repository contains code only. Point clouds, rasters, vector results,
> coordinates and screenshots of real project data never enter git or any web service. Read
> [DATA_POLICY.md](DATA_POLICY.md) before your first commit.

## Pipeline

```
point cloud ──► Stage 1 rasters ──► Stage 2 paint ──► Stage 3 elements ──► Stage 4 objects ──► Stage 5 GeoPackage
 (LAS/LAZ)      ground, intensity    probability       polygons, types,     tracks + s-intervals,  + QA, confidence,
                correction, density  (calibrated)      direction field      crosswalks, symbols    review queue
```

Design: [docs/handoff.md](docs/handoff.md). Step-by-step plan and status:
[docs/ROADMAP.md](docs/ROADMAP.md). Decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup (Windows / Linux, Python 3.12, [uv](https://docs.astral.sh/uv/))

```powershell
uv sync --extra ground                       # core + CSF ground filter
uv run pre-commit install --hook-type pre-commit --hook-type pre-push   # data guard
uv run roadmark init                         # data root: <repo>/data (git-ignored)
uv run pytest
```

**GPU machine** (model training/inference): `uv sync --extra ground --extra seg-cuda`
(PyTorch with CUDA 13.0; needs NVIDIA driver ≥ 580), then check with `uv run roadmark env`.
`--extra seg-cpu` installs CPU PyTorch for smoke tests elsewhere. Data moves between the
machines only as described in [DATA_POLICY.md](DATA_POLICY.md#compute-machines).

The data root defaults to `<repo>/data`. Set `ROADMARK_DATA_ROOT` (see `.env.example`) to use
another disk. PDAL is optional and used through its own CLI (conda/OSGeo4W) if needed.

## Try it on the synthetic scene

```powershell
uv run roadmark synth                         # writes <data_root>/synthetic/
uv run roadmark stage1 configs/pipeline/default.yaml configs/pipeline/synthetic.yaml
```

Outputs go to `<data_root>/runs/synthetic/stage1/`: COGs (`int_max`, `int_mean`, `int_p90`,
`density`, `coverage`, `hag_max`, `ground_z`, `road_mask`, `curb`), `qa_stage1.json`, `manifest.json`.

For real data, put site settings (inputs, EPSG, trajectory) in `configs/local/site.yaml`
(git-ignored) and run:

```powershell
uv run roadmark inspect data/raw/*.laz
uv run roadmark stage1 configs/pipeline/default.yaml configs/local/site.yaml
```

## Layout

```
configs/standards/   marking standard parameters (§11.2), draft until verified
configs/pipeline/    pipeline parameters; configs/local/ for site overrides (ignored)
src/roadmark/
  io/                LAS/LAZ and raster/COG I/O, shared grid
  stage1_raster/     ground filtering, intensity correction, channel rasters, road mask
  stage2_segment/    classical baseline, model, tiled inference, calibration
  stage3_elements/   polygonize, descriptors, direction field, fragment repair, glyphs
  stage4_assembly/   tracks, profiles, crosswalks, stop lines, symbols, areas
  stage5_vector/     GeoPackage schema and writer, exports
  qa/  eval/         rule checks, confidence, review; metrics vs gold standard
  synthetic.py       synthetic MLS scene with known truth
scripts/check_no_data.py   data-leak guard (pre-commit, pre-push, CI)
docs/                handoff, roadmap, decisions, annotation guidelines
```

## License

Code: [Apache License 2.0](LICENSE). The licence covers the code only and grants no rights to any
project data.
