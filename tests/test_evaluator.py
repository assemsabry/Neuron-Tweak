import pytest

torch = pytest.importorskip("torch")

from neuron_tweak.evaluator import (
    EvaluationError,
    ResponseMetrics,
    evaluate_refusal_rate,
    evaluate_responses,
    generate_responses,
)
from neuron_tweak.scorer import RefusalRateScorer


class FakeGenerationTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[-1]["content"]

    def __call__(self, texts, **kwargs):
        input_ids = torch.tensor([[1, 2, 3, 4] for _ in texts])
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, generated, skip_special_tokens):
        return [f"response-{index}" for index in range(len(generated))]


def test_refusal_rate_scorer_counts_hits():
    scorer = RefusalRateScorer()
    assert scorer.score(["I cannot help.", "Sure, okay."]) == 0.5
    assert scorer.score([]) == 0.0
    assert scorer.optimization == "minimize"

    custom = RefusalRateScorer(markers=["sure"])
    assert custom.score(["sure thing", "I cannot"]) == 0.5


def test_generate_responses_batches(runtime):
    tokenizer = FakeGenerationTokenizer()
    responses = generate_responses(
        runtime,
        tokenizer,
        ["a", "b", "c"],
        max_new_tokens=4,
        batch_size=2,
    )
    assert len(responses) == 3


def test_generate_responses_requires_prompts(runtime):
    from neuron_tweak.evaluator import generate_responses as _generate

    with pytest.raises(EvaluationError):
        _generate(runtime, FakeGenerationTokenizer(), [])


def test_evaluate_responses_returns_metrics(runtime):
    metrics = evaluate_responses(
        runtime,
        FakeGenerationTokenizer(),
        ["q1", "q2"],
        RefusalRateScorer(),
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
    )

    assert isinstance(metrics, ResponseMetrics)
    assert 0.0 <= metrics.refusal_rate <= 1.0
    assert metrics.mean_length > 0.0


def test_evaluate_refusal_rate_matches_metrics(runtime):
    tokenizer = FakeGenerationTokenizer()
    metrics = evaluate_responses(
        runtime,
        tokenizer,
        ["q1"],
        RefusalRateScorer(),
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
    )
    rate = evaluate_refusal_rate(
        runtime,
        tokenizer,
        ["q1"],
        RefusalRateScorer(),
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
    )
    assert rate == metrics.refusal_rate
