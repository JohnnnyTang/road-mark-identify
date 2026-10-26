"""Compute device selection and environment report.

Model code (Stage 2 segmentation, glyph CNN) runs on the NVIDIA GPU machine; everything else
runs anywhere. Training refuses to fall back to CPU silently.
"""

from __future__ import annotations

import platform

from roadmark.config import default_data_root


def get_device(require_cuda: bool = False):
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch missing: uv sync --extra seg-cuda (GPU machine) "
                          "or --extra seg-cpu (CPU smoke tests)") from e
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA required but not available. Run this on the GPU machine "
                           "(and check `uv run roadmark env`).")
    return torch.device("cpu")


def environment_report() -> dict:
    import rasterio

    root = default_data_root()
    rep = {"python": platform.python_version(), "platform": platform.platform(),
           "data_root": str(root), "data_root_exists": root.is_dir(),
           "gdal": rasterio.__gdal_version__}
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["torch_cuda_build"] = torch.version.cuda
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rep["gpus"] = [{"name": torch.cuda.get_device_name(i),
                            "memory_gb": round(torch.cuda.get_device_properties(i).total_memory
                                               / 2**30, 1)}
                           for i in range(torch.cuda.device_count())]
    except ImportError:
        rep["torch"] = None
    return rep
