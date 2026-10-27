"""Command line entry point: ``uv run roadmark --help``.

Output is redacted by default (sizes and counts, no absolute coordinates). See DATA_POLICY.md.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from roadmark.config import load_dotenv, load_pipeline, load_standard

app = typer.Typer(no_args_is_help=True, add_completion=False)


def _cfg(configs: list[Path]):
    return load_pipeline(*configs)


@app.callback()
def _main(verbose: bool = typer.Option(False, "-v", "--verbose")) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")


@app.command()
def inspect(files: list[Path],
            show_coords: bool = typer.Option(False, help="Print absolute bounds (local use only)")
            ) -> None:
    """Summarise LAS/LAZ headers: counts, density, extent size, CRS, available attributes."""
    from roadmark.io.pointcloud import read_header

    for f in files:
        typer.echo(json.dumps(read_header(f).summary(show_coords), indent=2))


@app.command()
def inventory(groups: list[str] = typer.Argument(
                  ..., help="NAME=GLOB (relative to the data root); repeat NAME to add globs"),
              name: str = typer.Option(..., help="Run name: writes runs/<name>/inventory/"),
              cell: float = 1.0, workers: int = 4) -> None:
    """Delivery inventory (Roadmap 1.1): per-group counts, classes, passes, density, intensity."""
    from roadmark.config import default_data_root
    from roadmark.io.inventory import InventoryCfg, run_inventory
    from roadmark.provenance import write_manifest

    cfg = InventoryCfg(groups=_group_spec(groups), cell=cell, workers=workers)
    root = default_data_root()
    rep, files = run_inventory(cfg, root)
    out = root / "runs" / name / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_manifest(out, cfg, files, {"stage": "inventory"})
    typer.echo(json.dumps({"groups": rep["groups"], "tiles": rep["tiles"]}, indent=2))


@app.command()
def raster_inventory(groups: list[str] = typer.Argument(
                         ..., help="NAME=GLOB relative to the data root; repeat NAME to add"),
                     name: str = typer.Option(..., help="Run name: writes runs/<name>/inventory/"),
                     workers: int = 4
                     ) -> None:
    """Delivered raster tiles (Roadmap 1.8): grid, dtype, no-data encoding, value percentiles."""
    from roadmark.config import default_data_root
    from roadmark.io.raster_inventory import RasterInventoryCfg, run_raster_inventory
    from roadmark.provenance import write_manifest

    cfg = RasterInventoryCfg(groups=_group_spec(groups), workers=workers)
    root = default_data_root()
    rep, files = run_raster_inventory(cfg, root)
    out = root / "runs" / name / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "raster_inventory.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_manifest(out, cfg, files, {"stage": "raster_inventory"})
    typer.echo(json.dumps({"groups": rep["groups"], "cross_group": rep["cross_group"]}, indent=2))


@app.command()
def crosscheck(channels: list[str] = typer.Argument(..., help="CHANNEL=GLOB of raster tiles"),
               points: list[str] = typer.Option([], help="Ground point globs (repeatable)"),
               name: str = typer.Option(..., help="Run name: writes runs/<name>/inventory/"),
               n_tiles: int = 12, workers: int = 3) -> None:
    """Raster tiles vs each other and vs the ground points: no-data encoding, registration."""
    from roadmark.config import default_data_root
    from roadmark.io.crosscheck import CrosscheckCfg, run_crosscheck
    from roadmark.provenance import write_manifest

    spec = {k: v[0] for k, v in _group_spec(channels).items()}
    cfg = CrosscheckCfg(channels=spec, points=points, n_tiles=n_tiles, workers=workers)
    root = default_data_root()
    rep, files = run_crosscheck(cfg, root)
    out = root / "runs" / name / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "crosscheck.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_manifest(out / "crosscheck", cfg, files, {"stage": "crosscheck"})
    typer.echo(json.dumps(rep, indent=2))


@app.command()
def probe_channels(channels: list[str] = typer.Argument(
                       ..., help="density=GLOB intensity=GLOB height=GLOB (raster tiles)"),
                   points: list[str] = typer.Option([], help="Ground point globs (repeatable)"),
                   name: str = typer.Option(..., help="Run name: writes runs/<name>/inventory/"),
                   n_tiles: int = 4, workers: int = 2) -> None:
    """Infer channel encodings: density units, intensity max/mean and scale, height meaning."""
    from roadmark.config import default_data_root
    from roadmark.io.channel_probe import ProbeCfg, run_probe
    from roadmark.provenance import write_manifest

    spec = {k: v[0] for k, v in _group_spec(channels).items()}
    cfg = ProbeCfg(channels=spec, points=points, n_tiles=n_tiles, workers=workers)
    root = default_data_root()
    rep, files = run_probe(cfg, root)
    out = root / "runs" / name / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "channel_probe.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_manifest(out / "channel_probe", cfg, files, {"stage": "probe_channels"})
    typer.echo(json.dumps(rep, indent=2))


@app.command()
def adapt_rasters(channels: list[str] = typer.Argument(
                      ..., help="density=GLOB intensity=GLOB (raster tiles); repeat to add globs"),
                  name: str = typer.Option(..., help="Run name: writes runs/<name>/stage1/"),
                  aggregate: int = typer.Option(1, help="k x k aggregation: count sum, max"),
                  coverage_min_points: int = 1, max_grid_mpx: float = 1500.0,
                  pick: int = typer.Option(
                      0, help="Pilot: use only the k x k neighbouring tiles with the most "
                              "occupied pixels (names go to runs/<name>/pilot_tiles.json)"),
                  workers: int = 4) -> None:
    """Delivered raster tiles -> I1 channel COGs (Roadmap 1.8): grid fix, decode, mosaic."""
    from roadmark.config import default_data_root
    from roadmark.provenance import write_manifest
    from roadmark.stage1_raster.raster_adapter import AdapterCfg, _glob, pick_pilot, run_adapter

    spec = _group_spec(channels)
    root = default_data_root()
    tiles, picked = None, None
    if pick:
        tiles, picked = pick_pilot(_glob(root, spec.get("density", [])), pick, workers)
        (root / "runs" / name).mkdir(parents=True, exist_ok=True)
        (root / "runs" / name / "pilot_tiles.json").write_text(
            json.dumps({"tiles": tiles, **picked}, indent=2), encoding="utf-8")
    cfg = AdapterCfg(channels=spec, aggregate=aggregate, tiles=tiles,
                     coverage_min_points=coverage_min_points, max_grid_mpx=max_grid_mpx)
    out = root / "runs" / name / "stage1"
    rep, files = run_adapter(cfg, root, out)
    rep["pilot_pick"] = picked
    (out / "qa_adapter.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_manifest(out, cfg, files, {"stage": "adapt_rasters", "channels": rep["channels"]})
    # The manifest config holds the tile names; the console shows counts only.
    typer.echo(json.dumps({k: rep[k] for k in ("pilot_pick", "grid", "tiles", "overlaps",
                                               "coverage")}, indent=2))


@app.command()
def vector_inventory(paths: list[str] = typer.Argument(
                         ..., help="Globs of vector files relative to the data root (** ok)"),
                     name: str = typer.Option(..., help="Run name: writes runs/<name>/inventory/"),
                     tiles: list[str] = typer.Option([], help="Raster tile globs (footprint)"),
                     grid: str = typer.Option(None, help="Stage 1 folder for registration, "
                                                         "e.g. runs/pilot_a/stage1"),
                     group_by: list[str] = typer.Option(
                         [], help="Code field(s): line profile per value (repeatable)")) -> None:
    """Annotation vectors (I3 intake): schema, sizes, and registration with the rasters."""
    from roadmark.config import default_data_root
    from roadmark.io.vector_inventory import VectorInventoryCfg, run_vector_inventory
    from roadmark.provenance import write_manifest

    cfg = VectorInventoryCfg(paths=paths, tiles=tiles, grid=grid, group_by=group_by)
    root = default_data_root()
    rep, names, files = run_vector_inventory(cfg, root)
    out = root / "runs" / name / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / "vector_inventory.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out / "vector_names.json").write_text(json.dumps(names, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    write_manifest(out / "vector_inventory", cfg, files, {"stage": "vector_inventory"})
    typer.echo(json.dumps(rep, indent=2, ensure_ascii=False))


def _group_spec(groups: list[str]) -> dict[str, list[str]]:
    spec: dict[str, list[str]] = {}
    for g in groups:
        key, _, pat = g.rpartition("=")
        spec.setdefault(key or "all", []).append(pat)
    return spec


@app.command()
def check_config(configs: list[Path]) -> None:
    """Validate pipeline config layers and the standards file they reference."""
    cfg = _cfg(configs)
    std = load_standard(cfg.standard)
    from roadmark.config import config_hash

    typer.echo(f"pipeline '{cfg.name}' ok (hash {config_hash(cfg)}); "
               f"standard '{std.standard} {std.edition}' verified={std.verified}")
    if not std.verified:
        typer.secho("WARNING: standard values are DRAFT; verify before production.", fg="yellow")


@app.command()
def init() -> None:
    """Create the data-root layout (default <repo>/data, or $ROADMARK_DATA_ROOT)."""
    from roadmark.datalayout import init_data_root

    root = init_data_root()
    typer.echo(f"data root ready: {root}")


@app.command()
def env() -> None:
    """Report this machine's environment: data root, GDAL, PyTorch/CUDA/GPU."""
    from roadmark.device import environment_report

    typer.echo(json.dumps(environment_report(), indent=2))


@app.command()
def synth(out: Path = typer.Option(None, help="Default: <data_root>/synthetic"),
          length: float = 60.0, density: float = 3000.0, seed: int = 0) -> None:
    """Generate the synthetic test scene (local coordinates) into the data root."""
    from roadmark.config import default_data_root
    from roadmark.synthetic import SceneParams, generate

    out = out or default_data_root() / "synthetic"
    counts = generate(out, SceneParams(length=length, density=density, seed=seed))
    typer.echo(json.dumps({"written_to": out.name, **counts}))


@app.command()
def stage1(configs: list[Path], keep_work: bool = False) -> None:
    """Stage 1: ground filtering, intensity correction, channel rasters, road mask, QA."""
    from roadmark.stage1_raster.pipeline import run_stage1

    rep = run_stage1(_cfg(configs), keep_work=keep_work)
    cp = rep["cross_pass_consistency"]
    typer.echo(json.dumps({"grid": rep["grid"], "intensity_model": rep["intensity_model"],
                           "cross_pass_consistency": cp, "coverage": rep["coverage"]}, indent=2))


if __name__ == "__main__":
    app()
