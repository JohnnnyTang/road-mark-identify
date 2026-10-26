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
