"""Accelerator detection and hardware reporting.

The analysis phases use this module to report the active backend before model
loading. Batch-size tuning will be added with the activation-capture pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendInfo:
    """A short description of the available PyTorch backend."""

    backend: str
    device_name: str
    total_memory_gb: float | None
    torch_version: str | None

    def summary(self) -> str:
        if self.backend == "unavailable":
            return "PyTorch is not installed"
        memory = (
            f", {self.total_memory_gb:.1f} GB"
            if self.total_memory_gb is not None
            else ""
        )
        return f"{self.backend} ({self.device_name}{memory})"


def detect_backend() -> BackendInfo:
    """Detect CUDA, ROCm, MPS, or CPU without requiring torch at import time."""
    try:
        import torch
    except ModuleNotFoundError:
        return BackendInfo(
            backend="unavailable",
            device_name="not installed",
            total_memory_gb=None,
            torch_version=None,
        )

    torch_version = getattr(torch, "__version__", None)

    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            properties = torch.cuda.get_device_properties(0)
            memory_gb = properties.total_memory / (1024**3)
        except Exception:  # noqa: BLE001 - keep the report best-effort
            device_name = "CUDA device"
            memory_gb = None
        backend = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        return BackendInfo(backend, device_name, memory_gb, torch_version)

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return BackendInfo("MPS", "Apple Silicon", None, torch_version)

    return BackendInfo("CPU", "no GPU backend detected", None, torch_version)
