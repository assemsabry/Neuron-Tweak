import json
from datetime import datetime

import pytest

from neuron_tweak.manifest import (
    MANIFEST_NAME,
    ManifestError,
    build_manifest,
    load_manifest,
    plan_from_manifest,
    write_manifest,
)
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


def test_manifest_round_trip_to_plan(tmp_path):
    edit = NeuronEdit(layer=2, neuron=5, delta=-1.0)
    results = [
        AppliedEdit(
            edit=edit,
            projection="down_proj",
            original_norm=1.0,
            new_norm=0.0,
        )
    ]
    path = write_manifest(
        tmp_path, build_manifest(_info(), "down_proj", results)
    )

    plan = plan_from_manifest(load_manifest(path))

    assert plan.to_text() == "L2 N5 -1"


def test_load_manifest_rejects_other_json(tmp_path):
    path = tmp_path / "other.json"
    path.write_text('{"tool": "something-else"}', encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(path)


def test_load_manifest_rejects_missing_file(tmp_path):
    with pytest.raises(ManifestError):
        load_manifest(tmp_path / "missing.json")


def test_plan_from_manifest_rejects_bad_entries():
    with pytest.raises(ManifestError):
        plan_from_manifest({"tool": "neuron-tweak", "edits": [{"layer": "x"}]})
    with pytest.raises(ManifestError):
        plan_from_manifest({"tool": "neuron-tweak", "edits": []})
