import pytest

from neuron_tweak.optimizer import NeuronStudy, OptimizerError, TrialResult


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


def test_neuron_study_supports_multi_objective():
    def objective(trial):
        top_k = trial.suggest_int("top_k", 1, 2)
        return (float(top_k), float(top_k) * 2.0)

    study = NeuronStudy(
        objective,
        n_trials=2,
        n_startup_trials=1,
        direction=("minimize", "minimize"),
    )

    results = study.run()

    assert len(results) == 2
    for result in results:
        assert isinstance(result.score, float)
        assert result.capability is not None
        assert result.capability >= result.score


def test_neuron_study_rejects_invalid_settings():
    with pytest.raises(OptimizerError):
        NeuronStudy(lambda trial: 0.0, n_trials=0)
    with pytest.raises(OptimizerError):
        NeuronStudy(lambda trial: 0.0, direction="sideways")
