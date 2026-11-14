"""Optuna-based neuron optimization study."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


class OptimizerError(RuntimeError):
    """Raised when an optimization study cannot run."""


@dataclass(frozen=True)
class TrialResult:
    """One completed Optuna trial."""

    trial_number: int
    params: dict[str, Any]
    score: float


class NeuronStudy:
    """Run Optuna trials over editable neuron parameters."""

    def __init__(
        self,
        objective: Callable[[Any], float],
        *,
        n_trials: int = 30,
        n_startup_trials: int = 10,
        direction: str = "minimize",
    ) -> None:
        if n_trials <= 0:
            raise OptimizerError("n_trials must be greater than zero")
        if n_startup_trials < 0:
            raise OptimizerError("n_startup_trials cannot be negative")
        if direction not in {"minimize", "maximize"}:
            raise OptimizerError(
                f"Invalid study direction {direction!r}; expected minimize or maximize"
            )
        self.objective = objective
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.direction = direction

    def run(self) -> list[TrialResult]:
        """Run the study and return the trial results in study order."""
        try:
            import optuna
        except ModuleNotFoundError as error:
            raise OptimizerError(
                "optuna is not installed. Install the project dependencies with "
                "'pip install -e .'"
            ) from error

        study = optuna.create_study(direction=self.direction)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        results: list[TrialResult] = []
        trials = sorted(study.trials, key=lambda trial: trial.number)
        for trial in trials:
            score = trial.value
            results.append(
                TrialResult(
                    trial_number=int(trial.number),
                    params=dict(trial.params or {}),
                    score=float(score) if score is not None else float("inf"),
                )
            )
        return results
