import json
from datetime import datetime

from neuron_tweak.manifest import MANIFEST_NAME, build_manifest, write_manifest
from neuron_tweak.model import AppliedEdit, ModelInfo
from neuron_tweak.neurons import NeuronEdit


def _info() -> ModelInfo:
    return ModelInfo(
        model_id="test/model",
        num_layers=2,
        hidden_size=16,
        intermediate_size=32,
        dtype="float32",
        device="cpu",
    )


def test_build_manifest_records_edits():
    edit = NeuronEdit(layer=1, neuron=3, delta=-1.0)
    results = [
        AppliedEdit(
            edit=edit,
            projection="down_proj",
            original_norm=1.5,
            new_norm=0.0,
        )
    ]

    manifest = build_manifest(_info(), "down_proj", results)

    assert manifest["tool"] == "neuron-tweak"
    assert manifest["model_id"] == "test/model"
    assert manifest["num_layers"] == 2
    assert manifest["target"] == "down_proj"
    assert manifest["edits"] == [
        {
            "layer": 1,
            "neuron": 3,
            "delta": -1.0,
            "scale": 0.0,
            "projection": "down_proj",
            "original_norm": 1.5,
            "new_norm": 0.0,
        }
    ]
    datetime.fromisoformat(manifest["created_at"])


def test_build_manifest_without_edits():
    manifest = build_manifest(_info(), "up_proj", [])

    assert manifest["edits"] == []
    assert manifest["target"] == "up_proj"


def test_write_manifest_roundtrips(tmp_path):
    path = write_manifest(tmp_path, {"tool": "neuron-tweak", "edits": []})

    assert path.name == MANIFEST_NAME
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "tool": "neuron-tweak",
        "edits": [],
    }
