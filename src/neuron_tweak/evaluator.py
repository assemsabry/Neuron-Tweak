"""Scorer orchestration, objective values, and response generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .scorer import DEFAULT_REFUSAL_MARKERS, RefusalRateScorer

if TYPE_CHECKING:
    from .model import ModelRuntime


class EvaluationError(RuntimeError):
    """Raised when model evaluation cannot run."""


@dataclass(frozen=True)
class ResponseMetrics:
    """Generated-response metrics for one evaluation pass."""

    refusal_rate: float
    mean_length: float


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


def generate_single_response(
    runtime: "ModelRuntime",
    tokenizer: Any | None,
    prompt: str,
    *,
    max_new_tokens: int = 128,
    max_length: int = 512,
) -> str:
    """Generate one response for a single user prompt."""
    effective_tokenizer = tokenizer if tokenizer is not None else runtime.tokenizer
    if effective_tokenizer is None:
        raise EvaluationError(
            "A tokenizer is required for response generation"
        )
    responses = generate_responses(
        runtime,
        effective_tokenizer,
        [prompt],
        max_new_tokens=max_new_tokens,
        batch_size=1,
        max_length=max_length,
    )
    return responses[0]


def evaluate_responses(
    runtime: "ModelRuntime",
    tokenizer: Any | None,
    prompts: Sequence[str],
    scorer: RefusalRateScorer,
    *,
    max_new_tokens: int = 64,
    batch_size: int = 4,
    max_length: int = 512,
) -> ResponseMetrics:
    """Generate responses and return refusal and length metrics."""
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
    mean_length = (
        sum(len(response.split()) for response in responses) / len(responses)
        if responses
        else 0.0
    )
    return ResponseMetrics(
        refusal_rate=scorer.score(responses),
        mean_length=mean_length,
    )


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
    metrics = evaluate_responses(
        runtime,
        tokenizer,
        prompts,
        scorer,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        max_length=max_length,
    )
    return metrics.refusal_rate

