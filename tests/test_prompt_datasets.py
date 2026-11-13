import re

import pytest

from neuron_tweak.prompts import DEFAULT_DATA_DIR, IDENTITY_PLACEHOLDERS, load_prompt_file

PROMPT_FILES = {
    "safety/good": DEFAULT_DATA_DIR / "safety" / "good_prompts.txt",
    "safety/bad": DEFAULT_DATA_DIR / "safety" / "bad_prompts.txt",
    "identity/good": DEFAULT_DATA_DIR / "identity" / "good_prompts.txt",
    "identity/bad": DEFAULT_DATA_DIR / "identity" / "bad_prompts.txt",
}


@pytest.mark.parametrize("name", sorted(PROMPT_FILES))
def test_prompt_file_has_600_unique_lines(name):
    prompts = load_prompt_file(PROMPT_FILES[name])
    assert len(prompts) == 600
    assert len(set(prompts)) == 600


@pytest.mark.parametrize("name", sorted(PROMPT_FILES))
def test_prompt_file_is_ascii(name):
    text = PROMPT_FILES[name].read_text(encoding="utf-8")
    assert all(ord(character) < 128 for character in text)


def test_identity_good_prompts_use_placeholders():
    prompts = load_prompt_file(PROMPT_FILES["identity/good"])
    pattern = re.compile(
        "|".join(r"\{" + placeholder + r"\}" for placeholder in IDENTITY_PLACEHOLDERS)
    )
    assert all(pattern.search(prompt) for prompt in prompts)


def test_identity_good_prompts_expand_to_target_identity():
    prompts = load_prompt_file(PROMPT_FILES["identity/good"])
    values = {
        "model_name": "AcmeLM",
        "vendor": "Acme Corporation",
        "owner": "Acme Corporation",
        "founder": "Jane Doe",
        "product_line": "Acme Assist",
        "assistant_persona": "Acme Assist, the official Acme assistant",
    }
    [first] = [
        prompt.format(**values) for prompt in prompts if "model_name" in prompt
    ][:1]
    assert "AcmeLM" in first
