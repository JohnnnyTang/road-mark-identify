"""Data-leak guard: refuse to commit or push anything that looks like project data.

Runs as a pre-commit hook (staged files), a pre-push hook and in CI (``--all``).
Exit code 1 blocks the commit/push and lists every offending file with the reason.

Checks, in order:
  1. path      - anything under a local data root (data/, runs/, models/, ...)
  2. extension - point clouds, rasters, vectors, arrays, weights, archives, images
  3. magic     - file content signature (catches renamed binaries: LAS, TIFF, SQLite/GPKG, ...)
  4. size      - any file larger than MAX_BYTES
  5. notebook  - .ipynb with stored outputs (outputs embed rendered data)
  6. coords    - text lines with absolute-coordinate-looking number pairs

A single line can opt out of check 6 with the pragma ``data-guard: allow``.
Images are allowed only under ALLOWED_IMAGE_DIRS, and only synthetic/cleared ones.

Standard library only, so it runs before ``uv sync`` and in bare CI.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 512 * 1024

BLOCKED_DIRS = (
    "data/", "runs/", "outputs/", "models/", "checkpoints/", "scratch/",
    "mlruns/", "lightning_logs/", "wandb/", "tb_logs/", "configs/local/",
)
ALLOWED_IN_BLOCKED_DIRS: set[str] = set()  # nothing under data roots is ever tracked

BLOCKED_EXT = {
    # point clouds / trajectories
    ".las", ".laz", ".e57", ".ply", ".pcd", ".pts", ".ptx", ".xyz", ".rcp", ".rcs",
    ".lasd", ".sbet", ".trj", ".traj",
    # rasters
    ".tif", ".tiff", ".vrt", ".img", ".jp2", ".ecw", ".sid", ".asc", ".nc", ".ovr",
    ".tfw", ".jgw", ".pgw",
    # vectors
    ".gpkg", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx", ".geojson",
    ".geojsonl", ".kml", ".kmz", ".gml", ".dxf", ".dwg", ".fgb", ".osm", ".pbf", ".sqlite",
    ".db", ".parquet", ".geoparquet", ".qgz",
    # arrays / models
    ".npy", ".npz", ".h5", ".hdf5", ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt",
    ".onnx", ".safetensors", ".memmap", ".dat",
    # archives
    ".zip", ".7z", ".rar", ".tar", ".gz", ".tgz", ".bz2", ".xz",
    # env
    ".env",
}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
ALLOWED_IMAGE_DIRS = ("docs/figures/public/",)

MAGIC = {
    b"LASF": "LAS/LAZ point cloud",
    b"II*\x00": "TIFF/GeoTIFF",
    b"MM\x00*": "TIFF/GeoTIFF",
    b"II+\x00": "BigTIFF",
    b"MM\x00+": "BigTIFF",
    b"SQLite format 3\x00": "SQLite/GeoPackage",
    b"\x89HDF\r\n\x1a\n": "HDF5",
    b"\x93NUMPY": "NumPy array",
    b"PK\x03\x04": "ZIP archive (also .pt/.kmz/.qgz)",
    b"7z\xbc\xaf\x27\x1c": "7z archive",
    b"\x1f\x8b": "gzip archive",
    b"ply\n": "PLY point cloud",
    b"PAR1": "Parquet",
    b"\x80\x04": "Python pickle",
    b"\x80\x05": "Python pickle",
    b"\x89PNG": "PNG image",
    b"\xff\xd8\xff": "JPEG image",
}

TEXT_EXT = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".csv", ".tsv",
            ".cfg", ".ini", ".wkt", ".sql", ".sh", ".ps1", ".ipynb"}
PRAGMA = "data-guard: allow"
# Two projected coordinates on one line (e.g. 512345.67 3456789.01, or GK with zone 39512345.6).
RE_PROJECTED = re.compile(r"\b\d{6,8}\.\d+\b[\s,;]+\b\d{6,8}\.\d+\b")
# Two high-precision geographic coordinates on one line (e.g. 31.234567, 121.456789).
RE_GEOGRAPHIC = re.compile(r"(?<![\d.])-?\d{1,3}\.\d{5,}\b[\s,;]+-?\d{1,3}\.\d{5,}\b")

SELF = "scripts/check_no_data.py"


def _norm(p: str) -> str:
    return str(PurePosixPath(p.replace("\\", "/")))


def check_file(rel: str, root: Path) -> list[str]:
    rel = _norm(rel)
    path = root / rel
    reasons: list[str] = []
    low = rel.lower()
    suffix = PurePosixPath(low).suffix

    if rel not in ALLOWED_IN_BLOCKED_DIRS and any(low.startswith(d) for d in BLOCKED_DIRS):
        reasons.append("path is inside a local data directory")
    if suffix in BLOCKED_EXT or low.endswith(".copc.laz") or PurePosixPath(low).name == ".env":
        reasons.append(f"blocked data extension '{suffix}'")
    if suffix in IMAGE_EXT and not any(low.startswith(d) for d in ALLOWED_IMAGE_DIRS):
        reasons.append(f"image outside {ALLOWED_IMAGE_DIRS} (images may show project data)")

    if not path.is_file():  # deleted in this commit: nothing more to inspect
        return reasons

    size = path.stat().st_size
    if size > MAX_BYTES:
        reasons.append(f"file is {size / 1024:.0f} KiB > {MAX_BYTES // 1024} KiB limit")

    with path.open("rb") as fh:
        head = fh.read(32)
    for sig, kind in MAGIC.items():
        if head.startswith(sig):
            image_ok = kind.endswith("image") and any(low.startswith(d) for d in ALLOWED_IMAGE_DIRS)
            if not image_ok:
                reasons.append(f"content signature looks like {kind}")
            break

    if suffix in TEXT_EXT and rel != SELF and size <= MAX_BYTES:
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".ipynb":
            try:
                nb = json.loads(text)
                if any(c.get("outputs") for c in nb.get("cells", [])):
                    reasons.append("notebook has stored outputs (run nbstripout)")
            except json.JSONDecodeError:
                reasons.append("unparseable notebook")
        for n, line in enumerate(text.splitlines(), 1):
            if PRAGMA in line:
                continue
            if RE_PROJECTED.search(line) or RE_GEOGRAPHIC.search(line):
                reasons.append(f"line {n} looks like absolute coordinates "
                               f"(use synthetic/local coords, or add '# {PRAGMA}')")
                break
    return reasons


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True)
    return [f for f in out.stdout.decode("utf-8").split("\0") if f]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="files to check (default: staged by pre-commit)")
    ap.add_argument("--all", action="store_true", help="check every tracked file")
    args = ap.parse_args(argv)

    root = Path.cwd()
    files = tracked_files(root) if args.all or not args.files else args.files
    failures = {f: r for f in files if (r := check_file(f, root))}
    if not failures:
        return 0
    print("DATA GUARD: refusing - these files look like project data (see DATA_POLICY.md):",
          file=sys.stderr)
    for f, reasons in failures.items():
        print(f"  {f}", file=sys.stderr)
        for r in reasons:
            print(f"      - {r}", file=sys.stderr)
    print("If a file is genuinely synthetic and safe, move it/allowlist it in a reviewed PR; "
          "never bypass with --no-verify or git add -f.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
