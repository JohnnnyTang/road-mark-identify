"""Run manifests: which code, config and inputs produced an output (§2.7)."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from roadmark.config import config_hash

_PKGS = ("roadmark", "numpy", "scipy", "laspy", "rasterio", "shapely", "geopandas", "torch")


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_fingerprint(path: Path, head_bytes: int = 1 << 20) -> str:
    """Cheap identity of a large input: size + sha256 of the first MiB."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        h.update(fh.read(head_bytes))
    return f"{path.stat().st_size}:{h.hexdigest()[:16]}"


def package_versions() -> dict[str, str]:
    out = {}
    for p in _PKGS:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def provenance_string(cfg: BaseModel, stage: str) -> str:
    """Compact string for the `provenance` field of vector outputs (§8.2)."""
    return f"roadmark={package_versions().get('roadmark', '?')};git={git_commit()};" \
           f"cfg={config_hash(cfg)};stage={stage}"


def write_manifest(run_dir: Path, cfg: BaseModel, inputs: list[Path], extra: dict | None = None
                   ) -> Path:
    # Input file *names* only, never their absolute location.
    manifest = {
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "config_hash": config_hash(cfg),
        "config": cfg.model_dump(mode="json"),
        "packages": package_versions(),
        "python": platform.python_version(),
        "inputs": {p.name: file_fingerprint(p) for p in inputs},
        **(extra or {}),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out
