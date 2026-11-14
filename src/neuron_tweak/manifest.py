"""Edit manifests written alongside saved models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .model import AppliedEdit, ModelInfo
from .neurons import DEFAULT_MAX_SCALE, NeuronEdit, NeuronPlan

MANIFEST_NAME = "neuron_tweak_manifest.json"


class ManifestError(RuntimeError):
    """Raised when a manifest cannot be loaded or applied."""


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


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate a manifest written by :func:`write_manifest`."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ManifestError(f"Manifest not found: {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ManifestError(
            f"Could not read {manifest_path}: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise ManifestError(
            f"Invalid manifest JSON in {manifest_path}: {error}"
        ) from error
    if not isinstance(data, dict) or data.get("tool") != "neuron-tweak":
        raise ManifestError(f"Not a Neuron Tweak manifest: {manifest_path}")
    if not isinstance(data.get("edits"), list):
        raise ManifestError(f"Manifest has no edits list: {manifest_path}")
    return data


def plan_from_manifest(
    manifest: dict[str, Any],
    max_scale: float = DEFAULT_MAX_SCALE,
) -> NeuronPlan:
    """Rebuild a neuron plan from a manifest's edit entries."""
    plan = NeuronPlan(max_scale=max_scale)
    for entry in manifest.get("edits", []):
        try:
            layer = int(entry["layer"])
            neuron = int(entry["neuron"])
            delta = float(entry["delta"])
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestError(
                f"Invalid manifest edit entry: {entry!r}"
            ) from error
        plan.add(
            NeuronEdit(
                layer=layer,
                neuron=neuron,
                delta=delta,
                max_scale=max_scale,
            )
        )
    if not plan:
        raise ManifestError("Manifest contains no neuron edits")
    return plan
