"""Edit manifests written alongside saved models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .model import AppliedEdit, ModelInfo

MANIFEST_NAME = "neuron_tweak_manifest.json"


def build_manifest(
    info: ModelInfo,
    target: str,
    results: Sequence[AppliedEdit],
) -> dict[str, Any]:
    """Build a serializable record of the applied neuron edits."""
    return {
        "tool": "neuron-tweak",
        "version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": info.model_id,
        "num_layers": info.num_layers,
        "hidden_size": info.hidden_size,
        "intermediate_size": info.intermediate_size,
        "dtype": info.dtype,
        "device": info.device,
        "target": target,
        "edits": [
            {
                "layer": result.edit.layer,
                "neuron": result.edit.neuron,
                "delta": result.edit.delta,
                "scale": result.edit.scale,
                "projection": result.projection,
                "original_norm": result.original_norm,
                "new_norm": result.new_norm,
            }
            for result in results
        ],
    }


def write_manifest(directory: str | Path, manifest: dict[str, Any]) -> Path:
    """Write the manifest JSON into a saved model directory."""
    path = Path(directory) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
