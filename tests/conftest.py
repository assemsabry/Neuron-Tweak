"""Shared test fixtures for the tiny Llama integration tests."""

import pytest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

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
    from neuron_tweak.model import ModelRuntime

    runtime = ModelRuntime(
        str(tiny_model_dir), dtype="float32", device_map="cpu"
    )
    runtime.load()
    yield runtime
    runtime.close()
