import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from neuron_tweak.model import ModelRuntimeError
from neuron_tweak.neurons import NeuronPlan


def test_model_info(runtime):
    info = runtime.info
    assert info.num_layers == 2
    assert info.hidden_size == 16
    assert info.intermediate_size == 32
    assert info.dtype == "float32"


def test_apply_plan_amplifies_and_deletes(runtime):
    weight = runtime.down_proj_weight(1)
    original_first = weight[:, 0].detach().clone()
    original_second = weight[:, 1].detach().clone()

    results = runtime.apply_plan(NeuronPlan.from_text("L1 N0 +1, L1 N1 -1"))

    assert len(results) == 2
    torch.testing.assert_close(weight[:, 0], original_first * 2)
    assert torch.count_nonzero(weight[:, 1]) == 0
    assert original_second.shape == weight[:, 1].shape


def test_reset_restores_original_weights(runtime):
    weight = runtime.down_proj_weight(2)
    original = weight[:, 5].detach().clone()

    runtime.apply_plan(NeuronPlan.from_text("L2 N5 -0.5"))
    assert not torch.allclose(weight[:, 5], original)

    restored = runtime.reset_edits()
    assert restored == 1
    torch.testing.assert_close(weight[:, 5], original)


def test_apply_replaces_previous_plan_from_original_weights(runtime):
    weight = runtime.down_proj_weight(1)
    original = weight[:, 3].detach().clone()

    runtime.apply_plan(NeuronPlan.from_text("L1 N3 +1"))
    torch.testing.assert_close(weight[:, 3], original * 2)

    runtime.apply_plan(NeuronPlan.from_text("L1 N3 +0.5"))
    torch.testing.assert_close(weight[:, 3], original * 1.5)


def test_no_change_edit_keeps_weights(runtime):
    weight = runtime.down_proj_weight(1)
    original = weight[:, 4].detach().clone()

    results = runtime.apply_plan(NeuronPlan.from_text("L1 N4 +0"))

    torch.testing.assert_close(weight[:, 4], original)
    assert results[0].original_norm == results[0].new_norm


def test_out_of_range_layer_is_rejected(runtime):
    with pytest.raises(ModelRuntimeError):
        runtime.validate_plan(NeuronPlan.from_text("L3 N0 +1"))


def test_out_of_range_neuron_is_rejected(runtime):
    with pytest.raises(ModelRuntimeError):
        runtime.validate_plan(NeuronPlan.from_text("L1 N32 +1"))


def test_save_writes_model_files(runtime, tmp_path):
    runtime.apply_plan(NeuronPlan.from_text("L1 N0 +1"))
    target = runtime.save(tmp_path / "edited-model")

    assert (target / "config.json").is_file()
    assert list(target.glob("*.safetensors"))


def test_apply_up_proj(runtime):
    weight = runtime.model.model.layers[0].mlp.up_proj.weight
    original = weight[0].detach().clone()

    runtime.apply_plan(NeuronPlan.from_text("L1 N0 +1"), target="up_proj")

    torch.testing.assert_close(weight[0], original * 2)


def test_apply_gate_proj_deletes_row(runtime):
    weight = runtime.model.model.layers[0].mlp.gate_proj.weight

    runtime.apply_plan(NeuronPlan.from_text("L1 N0 -1"), target="gate_proj")

    assert torch.count_nonzero(weight[0]) == 0


def test_apply_all_projections(runtime):
    mlp = runtime.model.model.layers[0].mlp
    up_original = mlp.up_proj.weight[0].detach().clone()
    gate_original = mlp.gate_proj.weight[0].detach().clone()
    down_original = mlp.down_proj.weight[:, 0].detach().clone()

    results = runtime.apply_plan(
        NeuronPlan.from_text("L1 N0 +1"), target="all"
    )

    assert len(results) == 3
    assert {result.projection for result in results} == {
        "up_proj",
        "gate_proj",
        "down_proj",
    }
    torch.testing.assert_close(mlp.up_proj.weight[0], up_original * 2)
    torch.testing.assert_close(mlp.gate_proj.weight[0], gate_original * 2)
    torch.testing.assert_close(mlp.down_proj.weight[:, 0], down_original * 2)


def test_reset_restores_all_projections(runtime):
    mlp = runtime.model.model.layers[0].mlp
    up_original = mlp.up_proj.weight[0].detach().clone()
    gate_original = mlp.gate_proj.weight[0].detach().clone()
    down_original = mlp.down_proj.weight[:, 0].detach().clone()

    runtime.apply_plan(NeuronPlan.from_text("L1 N0 -0.5"), target="all")
    restored = runtime.reset_edits()

    assert restored == 3
    torch.testing.assert_close(mlp.up_proj.weight[0], up_original)
    torch.testing.assert_close(mlp.gate_proj.weight[0], gate_original)
    torch.testing.assert_close(mlp.down_proj.weight[:, 0], down_original)


def test_unknown_target_is_rejected(runtime):
    with pytest.raises(ModelRuntimeError):
        runtime.validate_plan(NeuronPlan.from_text("L1 N0 +1"), target="mlp")

