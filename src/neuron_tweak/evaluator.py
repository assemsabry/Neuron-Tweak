"""Scorer orchestration, objective values, and response generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .scorer import DEFAULT_REFUSAL_MARKERS, RefusalRateScorer

if TYPE_CHECKING:
    from .model import ModelRuntime


class EvaluationError(RuntimeError):
    """Raised when model evaluation cannot run."""


def generate_responses(
    runtime: "ModelRuntime",
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    max_length: int = 512,
) -> list[str]:
    """Generate greedy responses for a list of prompts."""
    if not prompts:
        raise EvaluationError("No prompts were provided for generation")
    if batch_size <= 0 or max_new_tokens <= 0:
        raise EvaluationError("batch_size and max_new_tokens must be positive")

    try:
        import torch
    except ModuleNotFoundError as error:
        raise EvaluationError("PyTorch is not installed") from error

    try:
        device = runtime.model.device
    except AttributeError:
        device = next(runtime.model.parameters()).device

    responses: list[str] = []

    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        texts = []
        for prompt in batch:
            try:
                texts.append(
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:  # noqa: BLE001 - fall back to the raw prompt
                texts.append(prompt)

        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {
            key: value.to(device) for key, value in encoded.items()
        }
        generated = runtime.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        responses.extend(decoded)

    return responses


def evaluate_refusal_rate(
    runtime: "ModelRuntime",
    tokenizer: Any | None,
    prompts: Sequence[str],
    scorer: RefusalRateScorer,
    *,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    max_length: int = 512,
) -> float:
    """Generate responses and return the refusal rate."""
    effective_tokenizer = tokenizer if tokenizer is not None else runtime.tokenizer
    if effective_tokenizer is None:
        raise EvaluationError(
            "A tokenizer is required for response generation"
        )
    responses = generate_responses(
        runtime,
        effective_tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )
    return scorer.score(responses)

