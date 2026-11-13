import pytest

from neuron_tweak.prompts import (
    PromptDatasetError,
    expand_placeholders,
    load_prompt_file,
    load_workflow_prompts,
    split_prompts,
)


def test_load_prompt_file_ignores_comments_and_blank_lines(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("# comment\n\nFirst prompt\nSecond prompt\n", encoding="utf-8")
    assert load_prompt_file(path) == ["First prompt", "Second prompt"]


def test_load_prompt_file_rejects_missing_file(tmp_path):
    with pytest.raises(PromptDatasetError):
        load_prompt_file(tmp_path / "missing.txt")


def test_expand_placeholders_requires_all_values():
    with pytest.raises(PromptDatasetError):
        expand_placeholders(["{model_name}"], {"model_name": "AcmeLM"})


def test_expand_placeholders_replaces_values():
    values = {
        "model_name": "AcmeLM",
        "vendor": "Acme Corporation",
        "owner": "Acme Corporation",
        "founder": "Jane Doe",
        "product_line": "Acme Assist",
        "assistant_persona": "Acme Assist",
    }
    expanded = expand_placeholders(["I am {model_name} from {vendor}."], values)
    assert expanded == ["I am AcmeLM from Acme Corporation."]


def test_split_prompts_holds_out_last_items():
    prompts = [f"prompt {index}" for index in range(10)]
    split = split_prompts(prompts, evaluation_size=3)
    assert split.train == prompts[:7]
    assert split.evaluation == prompts[7:]


def test_split_prompts_is_reproducible_with_seed():
    prompts = [f"prompt {index}" for index in range(10)]
    first = split_prompts(prompts, evaluation_size=3, seed=42)
    second = split_prompts(prompts, evaluation_size=3, seed=42)
    assert first == second


def test_load_workflow_prompts_requires_identity_values():
    with pytest.raises(PromptDatasetError):
        load_workflow_prompts("identity")


def test_load_workflow_prompts_returns_two_splits():
    values = {
        "model_name": "AcmeLM",
        "vendor": "Acme Corporation",
        "owner": "Acme Corporation",
        "founder": "Jane Doe",
        "product_line": "Acme Assist",
        "assistant_persona": "Acme Assist",
    }
    good, bad = load_workflow_prompts("identity", identity_values=values)
    assert len(good.train) == 500
    assert len(good.evaluation) == 100
    assert len(bad.train) == 500
    assert len(bad.evaluation) == 100
    assert all("{" not in prompt for prompt in good.train)
