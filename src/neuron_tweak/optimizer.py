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
    capability: float | None = None


class NeuronStudy:
    """Run Optuna trials over editable neuron parameters."""

    def __init__(
        self,
        objective: Callable[[Any], Any],
        *,
        n_trials: int = 30,
        n_startup_trials: int = 10,
        direction: str | tuple[str, ...] = ("minimize",),
    ) -> None:
        directions = [direction] if isinstance(direction, str) else list(direction)
        if not directions:
            raise OptimizerError("A study needs at least one objective direction")
        if n_trials <= 0:
            raise OptimizerError("n_trials must be greater than zero")
        if n_startup_trials < 0:
            raise OptimizerError("n_startup_trials cannot be negative")
        for value in directions:
            if value not in {"minimize", "maximize"}:
                raise OptimizerError(
                    f"Invalid study direction {value!r}; expected minimize or maximize"
                )
        self.objective = objective
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.direction = tuple(directions)
        self.n_objectives = len(self.direction)

    def run(self) -> list[TrialResult]:
        """Run the study and return the trial results in study order."""
        try:
            import optuna
        except ModuleNotFoundError as error:
            raise OptimizerError(
                "optuna is not installed. Install the project dependencies with "
                "'pip install -e .'"
            ) from error

        study = optuna.create_study(directions=list(self.direction))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(self.objective, n_trials=self.n_trials)

        results: list[TrialResult] = []
        trials = sorted(study.trials, key=lambda trial: trial.number)
        for trial in trials:
            values: list[float] = []
            if len(self.direction) > 1:
                raw = trial.values
                if raw is not None:
                    values = [float(item) for item in raw]
            else:
                value = trial.value
                if value is not None:
                    values = [float(value)]
            score = values[0] if values else float("inf")
            capability = values[1] if len(values) > 1 else None
            results.append(
                TrialResult(
                    trial_number=int(trial.number),
                    params=dict(trial.params or {}),
                    score=score,
                    capability=capability,
                )
            )
        return results
