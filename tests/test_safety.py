import pytest

import torch

from neuron_tweak.analysis import NeuronScore
from neuron_tweak.evaluator import evaluate_refusal_rate
from neuron_tweak.optimizer import NeuronStudy, OptimizerError, TrialResult
from neuron_tweak.safety import SafetyNeuronObjective, run_safety_study
from neuron_tweak.scorer import RefusalRateScorer


def test_refusal_rate_scorer_counts_hits():
    scorer = RefusalRateScorer()
    assert scorer.score(["I cannot help.", "Sure, okay."]) == 0.5
    assert scorer.score([]) == 0.0
    assert scorer.optimization == "minimize"


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


def _make_neurons(count):
    return [
        NeuronScore(
            layer=1,
            neuron=index,
            mean_good=0.0,
            mean_bad=0.5,
            cohen_d=1.0,
            mean_diff=0.5,
        )
        for index in range(count)
    ]


def test_safety_objective_suppresses_top_neurons(runtime):
    objective = SafetyNeuronObjective(
        runtime,
        _make_neurons(3),
        ["q1", "q2"],
        scorer=RefusalRateScorer(),
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
        tokenizer=FakeGenerationTokenizer(),
    )

    class FakeTrial:
        params = {"top_k": 2, "delta": -1.0}

        def suggest_int(self, name, low, high, **kwargs):
            return int(self.params.get(name, low))

        def suggest_float(self, name, low, high, **kwargs):
            return float(self.params.get(name, low))

    rate = objective(FakeTrial())

    assert 0.0 <= rate <= 1.0
    assert torch.count_nonzero(runtime.down_proj_weight(1)[:, 0]) == 0
    assert torch.count_nonzero(runtime.down_proj_weight(1)[:, 1]) == 0
    assert torch.count_nonzero(runtime.down_proj_weight(1)[:, 2]) > 0


def test_safety_objective_scoring_uses_evaluator(runtime):
    scorer = RefusalRateScorer()
    rate = evaluate_refusal_rate(
        runtime,
        FakeGenerationTokenizer(),
        ["q1"],
        scorer,
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
    )
    assert 0.0 <= rate <= 1.0


class RefusingGenerationTokenizer(FakeGenerationTokenizer):
    def batch_decode(self, generated, skip_special_tokens):
        return ["I cannot help with that." for _ in generated]


def test_safety_objective_includes_capability_penalty(runtime):
    objective = SafetyNeuronObjective(
        runtime,
        _make_neurons(2),
        ["q1"],
        scorer=RefusalRateScorer(),
        max_new_tokens=4,
        batch_size=2,
        max_length=64,
        tokenizer=RefusingGenerationTokenizer(),
        heldout_good_prompts=["g1"],
        capability_weight=0.5,
        baseline_good_length=5.0,
    )

    class FakeTrial:
        params = {"top_k": 1, "delta": -1.0}

        def suggest_int(self, name, low, high, **kwargs):
            return int(self.params.get(name, low))

        def suggest_float(self, name, low, high, **kwargs):
            return float(self.params.get(name, low))

    # Both bad and good responses refuse, so the score is 1.0 + 0.5 * 1.0.
    assert objective(FakeTrial()) == pytest.approx(1.5)


def test_run_safety_study_returns_best(runtime):
    from rich.console import Console
    from types import SimpleNamespace

    args = SimpleNamespace(
        study=True,
        study_trials=2,
        study_startup_trials=1,
        study_direction="minimize",
        study_apply_best=False,
        save_directory=None,
        yes=True,
        eval_max_new_tokens=4,
        eval_batch_size=2,
        max_sequence_length=64,
        target="down_proj",
        max_scale=10.0,
    )

    best = run_safety_study(
        runtime,
        Console(),
        args,
        [_make_neurons(3)],
        _make_neurons(3),
        ["q1", "q2"],
        FakeGenerationTokenizer(),
    )

    assert best is not None
    assert isinstance(best, TrialResult)
    assert best.score >= 0.0


def test_neuron_study_runs_trials():
    def objective(trial):
        top_k = trial.suggest_int("top_k", 1, 3)
        delta = trial.suggest_float("delta", -1.0, 1.0)
        return -float(top_k) - float(delta)

    study = NeuronStudy(
        objective,
        n_trials=3,
        n_startup_trials=1,
        direction="minimize",
    )

    results = study.run()

    assert len(results) == 3
    assert all(isinstance(result, TrialResult) for result in results)
    best = min(results, key=lambda result: result.score)
    assert best.score < 0.0


def test_neuron_study_rejects_invalid_settings():
    with pytest.raises(OptimizerError):
        NeuronStudy(lambda trial: 0.0, n_trials=0)
    with pytest.raises(OptimizerError):
        NeuronStudy(lambda trial: 0.0, direction="sideways")
