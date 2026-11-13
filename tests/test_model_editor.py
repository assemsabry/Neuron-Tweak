import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from neuron_tweak.model import ModelRuntime, ModelRuntimeError
from neuron_tweak.neurons import NeuronPlan


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-llama")
    config = transformers.LlamaConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    model = transformers.LlamaForCausalLM(config)
    model.save_pretrained(path)
    return path


@pytest.fixture
def runtime(tiny_model_dir):
    runtime = ModelRuntime(str(tiny_model_dir), dtype="float32", device_map="cpu")
    runtime.load()
    yield runtime
    runtime.close()


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
