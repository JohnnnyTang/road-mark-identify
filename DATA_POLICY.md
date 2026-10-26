# Data Policy

This repository is **public**. The code is open; **the data is not**. Point clouds, rasters
and vector results of real roads are survey data of a real place. They may also be regulated
geographic information (for data captured in China: the Surveying and Mapping Law and the rules
on high-precision geographic data and its export). They never leave the machines they were
delivered to, unless the data owner gives written permission.

## What counts as data

Anything derived from the project's real measurements, including:

- raw and processed point clouds, trajectories, scanner positions
- rasters (intensity, density, probability maps, masks), even single tiles or crops
- vector outputs, annotations, gold standard, review exports
- screenshots, example crops, plots or notebook outputs that show real data
- **absolute coordinates, extents, bounding boxes, place names or site IDs** of the project area
- model weights trained on the data (they can memorise it), embeddings, caches
- statistics detailed enough to reconstruct the above

What **may** be committed: code, generic configs, the standards configuration (dimensional
parameters from public standards, never the standard's text), and **synthetic** fixtures generated
in local coordinates (e.g. 0–100 m).

## Where data lives

- Outside the repo by preference: set `ROADMARK_DATA_ROOT` in `.env` (see `.env.example`).
- Otherwise in `./data/`, which is git-ignored.
- Site-specific configs (real paths, extents, EPSG of the area) go in `configs/local/`
  (git-ignored), e.g. `configs/local/site.yaml`:

  ```yaml
  name: site-a
  crs_epsg: 4547                      # the project's projected CRS
  stage1:
    inputs: ["raw/*.laz"]
    ground: {method: csf}
    trajectory: {path: raw/trajectory.csv}
  ```

Layout under the data root:

```
raw/            delivered point clouds, trajectories (read-only, never modified)
synthetic/      output of `roadmark synth`
annotations/    training masks, vector gold standard (Step 2)
runs/<name>/    pipeline outputs per run (stage1/, stage2/, ...), each with manifest.json
models/         trained weights
```

## Enforcement layers

| Layer | Mechanism | Blocks |
|---|---|---|
| 1 | `.gitignore` | data dirs, data extensions, archives, weights, `.env` |
| 2 | `scripts/check_no_data.py` as **pre-commit** hook | staged files by path, extension, content signature (renamed binaries), size > 512 KiB, notebook outputs, coordinate-like text |
| 3 | same script as **pre-push** hook (`--all`) | anything that slipped past layer 2 |
| 4 | GitHub Actions `data-guard` job | PRs from other clones / contributors |
| 5 | `.claude/settings.json` + `CLAUDE.md` | AI assistant reading data files, uploading, force-adding, pushing without asking |
| 6 | Runtime | CLIs print redacted summaries (sizes, counts; no absolute coordinates) unless `--show-coords` is passed; outputs are only written under the data root |

## Rules for people

1. Run once per clone: `uv run pre-commit install --hook-type pre-commit --hook-type pre-push`.
2. Never `git add -f` / `--force`, never `--no-verify`.
3. No data in issues, PRs, discussions, gists, wikis, release assets, or chat tools.
4. No remote experiment trackers or hubs with data (W&B cloud, HF Hub uploads, cloud buckets).
   Track experiments locally (TensorBoard / MLflow file store under the data root).
5. Figures for docs: synthetic only, placed in `docs/figures/public/`, reviewed in a PR.
6. Downloading public pretrained weights or datasets is fine; uploading anything is not.

## Rules for AI assistants

See `CLAUDE.md`, "Data protection". Short version: do not open data files; work through the
CLI's redacted summaries; never send data or coordinates to any web service, artifact,
document, gist or issue; ask before every `git push`.

## If something leaks

1. Stop. Don't push more commits on top of it.
2. If it was only committed locally: `git reset` the commit and delete the file from the index.
3. If it reached GitHub: make the repository private immediately, then rewrite history
   (`git filter-repo`), force-push, and contact GitHub Support to purge cached views and forks.
   Tell the data owner. Assume the data was copied.
