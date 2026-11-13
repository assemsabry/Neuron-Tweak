"""Prompt dataset loading, placeholder expansion, and train/evaluation split."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

PROFILE_FILES: dict[str, dict[str, Path]] = {
    "safety": {
        "good": DEFAULT_DATA_DIR / "safety" / "good_prompts.txt",
        "bad": DEFAULT_DATA_DIR / "safety" / "bad_prompts.txt",
    },
    "identity": {
        "good": DEFAULT_DATA_DIR / "identity" / "good_prompts.txt",
        "bad": DEFAULT_DATA_DIR / "identity" / "bad_prompts.txt",
    },
}

PROFILE_ALIASES: dict[str, str] = {
    "refusal": "safety",
}

IDENTITY_PLACEHOLDERS: tuple[str, ...] = (
    "model_name",
    "vendor",
    "owner",
    "founder",
    "product_line",
    "assistant_persona",
)


class PromptDatasetError(ValueError):
    """Raised when a prompt dataset is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class PromptSplit:
    """A dataset split into localization/optimization and evaluation prompts."""

    train: list[str]
    evaluation: list[str]


def load_prompt_file(path: str | Path) -> list[str]:
    """Load prompts from a UTF-8 text file, one prompt per line.

    Empty lines and lines starting with "#" are ignored.
    """
    prompt_path = Path(path)
    if not prompt_path.is_file():
        raise PromptDatasetError(f"Prompt file not found: {prompt_path}")

    prompts: list[str] = []
    for line_number, raw_line in enumerate(
        prompt_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)

    if not prompts:
        raise PromptDatasetError(f"Prompt file is empty: {prompt_path}")
    return prompts


def expand_placeholders(
    prompts: list[str],
    values: dict[str, str],
) -> list[str]:
    """Expand identity placeholders such as {model_name} in each prompt."""
    missing = [name for name in IDENTITY_PLACEHOLDERS if name not in values]
    if missing:
        raise PromptDatasetError(
            "Missing target identity values: " + ", ".join(sorted(missing))
        )

    expanded: list[str] = []
    for prompt in prompts:
        try:
            expanded.append(prompt.format(**values))
        except KeyError as error:
            raise PromptDatasetError(
                f"Unknown placeholder {error} in prompt: {prompt!r}"
            ) from error
    return expanded


def split_prompts(
    prompts: list[str],
    evaluation_size: int = 100,
    seed: int | None = None,
) -> PromptSplit:
    """Split prompts into a train set and a held-out evaluation set.

    When seed is None, the last evaluation_size prompts are held out. When a
    seed is provided, the prompts are shuffled first, which is useful when the
    file is ordered by category.
    """
    if evaluation_size <= 0:
        raise PromptDatasetError("evaluation_size must be greater than zero")
    if evaluation_size >= len(prompts):
        raise PromptDatasetError(
            "evaluation_size must be smaller than the number of prompts"
        )

    ordered = list(prompts)
    if seed is not None:
        random.Random(seed).shuffle(ordered)

    return PromptSplit(
        train=ordered[:-evaluation_size],
        evaluation=ordered[-evaluation_size:],
    )


def load_workflow_prompts(
    profile: str,
    data_dir: str | Path | None = None,
    identity_values: dict[str, str] | None = None,
    evaluation_size: int = 100,
    seed: int | None = 0,
) -> tuple[PromptSplit, PromptSplit]:
    """Load the good and bad prompt sets for a workflow.

    Returns a tuple of (good_split, bad_split). Identity good prompts are
    expanded with identity_values before splitting.
    """
    if profile not in PROFILE_FILES:
        profile = PROFILE_ALIASES.get(profile, profile)
    if profile not in PROFILE_FILES:
        raise PromptDatasetError(
            f"Unknown profile {profile!r}; expected one of {sorted(PROFILE_FILES)}"
        )

    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    files = {
        kind: base_dir / profile / f"{kind}_prompts.txt"
        for kind in ("good", "bad")
    }

    good = load_prompt_file(files["good"])
    bad = load_prompt_file(files["bad"])

    if profile == "identity":
        if identity_values is None:
            raise PromptDatasetError(
                "Identity prompts require target identity values"
            )
        good = expand_placeholders(good, identity_values)

    return (
        split_prompts(good, evaluation_size=evaluation_size, seed=seed),
        split_prompts(bad, evaluation_size=evaluation_size, seed=seed),
    )
