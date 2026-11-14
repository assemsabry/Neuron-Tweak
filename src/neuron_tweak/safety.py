"""Safety Tweak: contrastive neuron localization and suppression.

The first Safety Tweak pipeline is deliberately simple:

1. Run the 500 training good prompts and 500 training bad prompts through the
   model and record MLP neuron activations.
2. Score every neuron with Cohen's d, positive meaning more active on bad
   prompts.
3. Select the top neurons per layer.
4. Suppress them with a Custom Tweak plan (default: delete, delta -1).

The 100 held-out evaluation prompts are split and reported, but automated
before/after generation scoring is a later phase.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, Sequence

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from .analysis import (
    AnalysisError,
    NeuronScore,
    capture_activations,
    global_top_neurons,
    plan_from_scores,
    score_neurons,
    top_neurons,
)
from .custom import render_applied, render_plan, save_plan_to
from .evaluator import evaluate_refusal_rate, evaluate_responses
from .manifest import build_manifest, write_manifest
from .optimizer import NeuronStudy, TrialResult
from .prompts import PromptDatasetError, load_workflow_prompts
from .scorer import IdentityMatchScorer, RefusalRateScorer

if TYPE_CHECKING:
    from rich.console import Console

    from .model import ModelRuntime


def parse_layer_selection(
    value: str | None, num_layers: int
) -> list[int] | None:
    """Parse 'all', '1,2,3', or '10-20' into 1-based layer numbers."""
    if value is None or not value.strip() or value.strip().lower() == "all":
        return None

    layers: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if start > end:
                    raise ValueError
                layers.update(range(start, end + 1))
            else:
                layers.add(int(token))
        except ValueError as error:
            raise AnalysisError(
                f"Invalid layer selection {value!r}; use 'all', '1,2,3', or "
                "'10-20'"
            ) from error

    invalid = [layer for layer in layers if layer < 1 or layer > num_layers]
    if invalid:
        raise AnalysisError(
            f"Layer selection {value!r} is outside 1..{num_layers}"
        )
    return sorted(layers)


def render_score_summary(
    console: "Console",
    scores_by_layer: Sequence[Sequence[NeuronScore]],
    selected: Sequence[NeuronScore],
    *,
    min_score: float,
    title: str = "Safety neuron localization",
) -> None:
    """Print a compact per-layer summary and the global top neurons."""
    by_layer: dict[int, list[NeuronScore]] = {}
    for score in selected:
        by_layer.setdefault(score.layer, []).append(score)

    table = Table(title=title)
    table.add_column("Layer", justify="right")
    table.add_column("Selected", justify="right")
    table.add_column("Max Cohen d", justify="right")
    table.add_column("Mean Cohen d", justify="right")

    for layer in sorted(by_layer):
        layer_scores = by_layer[layer]
        table.add_row(
            str(layer),
            str(len(layer_scores)),
            f"{max(score.cohen_d for score in layer_scores):.3f}",
            f"{sum(score.cohen_d for score in layer_scores) / len(layer_scores):.3f}",
        )
    console.print(table)

    top = global_top_neurons(
        scores_by_layer, count=min(20, len(selected)), min_score=min_score
    )
    if top:
        global_table = Table(title="Highest-scoring neurons")
        global_table.add_column("Neuron", style="cyan")
        global_table.add_column("Mean good", justify="right")
        global_table.add_column("Mean bad", justify="right")
        global_table.add_column("Delta mean", justify="right")
        global_table.add_column("Cohen d", justify="right")
        for score in top:
            global_table.add_row(
                score.address,
                f"{score.mean_good:.4f}",
                f"{score.mean_bad:.4f}",
                f"{score.mean_diff:+.4f}",
                f"{score.cohen_d:.3f}",
            )
        console.print(global_table)


def _capture(
    runtime: "ModelRuntime",
    console: "Console",
    prompts: Sequence[str],
    *,
    label: str,
    batch_size: int,
    max_length: int,
    layers: Sequence[int] | None,
):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Capturing {label} activations", total=len(prompts))
        return capture_activations(
            runtime,
            runtime.tokenizer,
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            layer_numbers=layers,
            label=label,
            progress=lambda done, total: progress.update(task, completed=done),
        )


class SafetyNeuronObjective:
    """Run one Optuna trial by varying top-k and the steering delta."""

    def __init__(
        self,
        runtime: "ModelRuntime",
        neurons: Sequence[NeuronScore],
        heldout_bad_prompts: Sequence[str],
        *,
        scorer: RefusalRateScorer | IdentityMatchScorer | None = None,
        max_new_tokens: int = 64,
        batch_size: int = 4,
        max_length: int = 512,
        target: str = "down_proj",
        max_scale: float = 10.0,
        tokenizer: Any | None = None,
        heldout_good_prompts: Sequence[str] | None = None,
        capability_weight: float = 0.0,
        baseline_good_length: float | None = None,
        goal: float = 1.0,
    ) -> None:
        self.runtime = runtime
        self.neurons = list(neurons)
        self.heldout_bad_prompts = list(heldout_bad_prompts)
        self.heldout_good_prompts = list(heldout_good_prompts or [])
        self.scorer = scorer or RefusalRateScorer()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.max_length = max_length
        self.target = target
        self.max_scale = max_scale
        self.capability_weight = capability_weight
        self.baseline_good_length = baseline_good_length
        self.goal = goal

    def __call__(self, trial: Any) -> float | tuple[float, float]:
        if hasattr(trial, "params"):
            params = trial.params or {}
        else:
            params = {}

        top_k = int(params.get("top_k", 1))
        delta = float(params.get("delta", -1.0))
        if hasattr(trial, "suggest_int"):
            top_k = int(trial.suggest_int("top_k", 1, max(1, len(self.neurons))))
        if hasattr(trial, "suggest_float"):
            delta = float(trial.suggest_float("delta", -1.0, 1.0))

        selected = self.neurons[:top_k]
        plan = plan_from_scores(selected, delta=delta, max_scale=self.max_scale)
        self.runtime.apply_plan(plan, target=self.target)
        bad_metric = evaluate_refusal_rate(
            self.runtime,
            self.tokenizer,
            self.heldout_bad_prompts,
            self.scorer,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            max_length=self.max_length,
        )
        bad_value = self.goal * bad_metric
        capability_value = 0.0
        if self.heldout_good_prompts and self.capability_weight > 0.0:
            good = evaluate_responses(
                self.runtime,
                self.tokenizer,
                self.heldout_good_prompts,
                self.scorer,
                max_new_tokens=self.max_new_tokens,
                batch_size=self.batch_size,
                max_length=self.max_length,
            )
            capability_value = self.capability_weight * (self.goal * good.refusal_rate)
            if self.baseline_good_length and good.mean_length > 0:
                capability_value += self.capability_weight * max(
                    0.0, 1.0 - good.mean_length / self.baseline_good_length
                )
        if not self.heldout_good_prompts or self.capability_weight <= 0.0:
            return bad_value
        return bad_value, capability_value


def run_safety_study(
    runtime: "ModelRuntime",
    console: "Console",
    args: object,
    scores: Sequence[Sequence[NeuronScore]],
    selected: Sequence[NeuronScore],
    heldout_bad_prompts: Sequence[str],
    tokenizer: Any | None = None,
    heldout_good_prompts: Sequence[str] | None = None,
    workflow: str = "Safety",
    scorer: RefusalRateScorer | IdentityMatchScorer | None = None,
    goal: float = 1.0,
) -> TrialResult | None:
    """Run the Optuna contrastive study and optionally apply the best trial."""
    from .optimizer import NeuronStudy

    selected_scorer = scorer or RefusalRateScorer()
    capability_weight = float(getattr(args, "study_capability_weight", 0.0))
    baseline_good_length: float | None = None
    if heldout_good_prompts and capability_weight > 0.0:
        baseline = evaluate_responses(
            runtime,
            tokenizer,
            heldout_good_prompts,
            selected_scorer,
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_batch_size,
            max_length=args.max_sequence_length,
        )
        baseline_good_length = baseline.mean_length
        console.print(
            f"* Baseline good-prompt score {baseline.refusal_rate:.3f}, "
            f"mean length {baseline.mean_length:.1f} words"
        )

    objective = SafetyNeuronObjective(
        runtime,
        selected,
        heldout_bad_prompts,
        max_new_tokens=args.eval_max_new_tokens,
        batch_size=args.eval_batch_size,
        max_length=args.max_sequence_length,
        target=args.target,
        max_scale=args.max_scale,
        tokenizer=tokenizer,
        heldout_good_prompts=heldout_good_prompts,
        capability_weight=capability_weight,
        baseline_good_length=baseline_good_length,
        scorer=selected_scorer,
        goal=goal,
    )
    direction = getattr(args, "study_direction", None) or (
        "maximize" if goal < 0 else "minimize"
    )
    study = NeuronStudy(
        objective,
        n_trials=args.study_trials,
        n_startup_trials=args.study_startup_trials,
        direction=direction,
    )
    console.print()
    console.print(
        f"[bold]Running {workflow.lower()} Optuna study: "
        f"{study.n_trials} trials[/]"
    )
    results = study.run()
    best = results[0]
    for result in results[1:]:
        result_capability = (
            float(result.capability) if result.capability is not None else float("inf")
        )
        best_capability = (
            float(best.capability) if best.capability is not None else float("inf")
        )
        if direction == "minimize":
            is_better = (
                result.score < best.score
                or (
                    result.score == best.score
                    and result_capability < best_capability
                )
            )
        else:
            is_better = (
                result.score > best.score
                or (
                    result.score == best.score
                    and result_capability < best_capability
                )
            )
        if is_better:
            best = result
    ranked = sorted(
        results,
        key=lambda result: (
            float(result.score),
            float(result.capability) if result.capability is not None else float("inf"),
        ),
        reverse=direction == "maximize",
    )

    table = Table(title=f"Optuna {workflow.lower()} study")
    table.add_column("Trial", justify="right")
    table.add_column("Top neurons", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Capability", justify="right")
    for result in ranked:
        capability = (
            float(result.capability)
            if result.capability is not None
            else float("inf")
        )
        table.add_row(
            str(result.trial_number),
            str(int(result.params.get("top_k", 1))),
            f"{float(result.params.get('delta', -1.0)):+.2f}",
            f"{float(result.score):.4f}",
            f"{capability:.4f}",
        )
    console.print(table)
    best_capability = (
        float(best.capability) if best.capability is not None else float("inf")
    )
    console.print(
        f"[green]Best trial: {best.trial_number} with objective score "
        f"{best.score:.4f} (capability penalty {best_capability:.4f})[/]"
    )

    chosen = best
    if not getattr(args, "study_apply_best", False):
        if not sys.stdin.isatty():
            return best
        import questionary

        choices = [
            questionary.Choice(
                title=(
                    f"Best: trial {best.trial_number} "
                    f"(top_k={best.params.get('top_k', 1)}, "
                    f"delta={best.params.get('delta', -1.0):+.2f}, "
                    f"score={best.score:.4f})"
                ),
                value=best,
            )
        ]
        for result in ranked[:9]:
            if result.trial_number == best.trial_number:
                continue
            choices.append(
                questionary.Choice(
                    title=f"Trial {result.trial_number} (score={result.score:.4f})",
                    value=result,
                )
            )
        chosen = questionary.select(
            f"Select a {workflow.lower()} study trial to apply:",
            choices=choices,
        ).ask()
        if chosen is None:
            return best

    top_k = int(chosen.params.get("top_k", 1))
    delta = float(chosen.params.get("delta", -1.0))
    plan = plan_from_scores(selected[:top_k], delta=delta, max_scale=args.max_scale)
    console.print()
    render_plan(console, plan, args.target)
    save_plan_to(console, plan, getattr(args, "save_plan", None))
    if not args.yes and sys.stdin.isatty():
        import questionary

        confirmed = questionary.confirm(
            f"Apply {workflow.lower()} trial {chosen.trial_number} to the model?",
            default=True,
        ).ask()
        if not confirmed:
            console.print("Edits were not applied.")
            return
    results_applied = runtime.apply_plan(plan, target=args.target)
    console.print()
    render_applied(console, results_applied)
    if getattr(args, "save_directory", None):
        saved_path = runtime.save(args.save_directory)
        manifest_path = write_manifest(
            saved_path,
            build_manifest(runtime.info, args.target, results_applied),
        )
        console.print(f"[green]Saved edited model to {saved_path}[/]")
        console.print(f"* Wrote edit manifest to {manifest_path}")
        console.print(f"* Saved model verified: {runtime.verify_saved(saved_path, results_applied)}")
    else:
        console.print(
            "[yellow]The edit is applied in memory only. No model directory "
            "was written.[/]"
        )


def run_safety_pipeline(
    runtime: "ModelRuntime",
    console: "Console",
    args: object,
    *,
    profile: str = "safety",
    identity_values: dict[str, str] | None = None,
) -> None:
    """Localize neurons, build a suppression plan, apply, and save."""
    workflow = "Identity" if profile == "identity" else "Safety"
    try:
        good_split, bad_split = load_workflow_prompts(
            profile,
            identity_values=identity_values,
            evaluation_size=args.evaluation_size,
            seed=args.seed,
        )
    except PromptDatasetError as error:
        raise AnalysisError(str(error)) from error

    layers = parse_layer_selection(args.layers, runtime.info.num_layers)
    good_prompts = good_split.train
    bad_prompts = bad_split.train
    limit = getattr(args, "analysis_prompts", 0)
    if limit > 0:
        good_prompts = good_prompts[:limit]
        bad_prompts = bad_prompts[:limit]

    console.print(
        f"* Localization prompts: {len(good_prompts)} good, "
        f"{len(bad_prompts)} bad"
    )
    console.print(
        f"* Held-out evaluation prompts: {len(good_split.evaluation)} good, "
        f"{len(bad_split.evaluation)} bad"
    )
    if layers is not None:
        console.print(f"* Selected layers: {', '.join(map(str, layers))}")

    good_stats = _capture(
        runtime,
        console,
        good_prompts,
        label="good",
        batch_size=args.analysis_batch_size,
        max_length=args.max_sequence_length,
        layers=layers,
    )
    bad_stats = _capture(
        runtime,
        console,
        bad_prompts,
        label="bad",
        batch_size=args.analysis_batch_size,
        max_length=args.max_sequence_length,
        layers=layers,
    )

    scores = score_neurons(good_stats, bad_stats)
    selected = top_neurons(
        scores,
        per_layer=args.top_neurons,
        layers=layers,
        min_score=args.min_score,
    )
    if not selected:
        console.print(
            "[yellow]No neurons passed the score threshold. Nothing to "
            "edit.[/]"
        )
        return

    if profile == "identity" and identity_values:
        scorer: RefusalRateScorer | IdentityMatchScorer = IdentityMatchScorer(
            [
                identity_values.get("model_name", ""),
                identity_values.get("vendor", ""),
                identity_values.get("owner", ""),
                identity_values.get("founder", ""),
                identity_values.get("product_line", ""),
            ]
        )
        metric_label = "Identity match rate"
        goal = -1.0
    else:
        scorer = RefusalRateScorer()
        metric_label = "Refusal rate"
        goal = 1.0

    if getattr(args, "study", False):
        run_safety_study(
            runtime,
            console,
            args,
            scores,
            selected,
            bad_split.evaluation,
            runtime.tokenizer,
            heldout_good_prompts=good_split.evaluation,
            workflow=workflow,
            scorer=scorer,
            goal=goal,
        )
        return

    console.print()
    render_score_summary(
        console,
        scores,
        selected,
        min_score=args.min_score,
        title=f"{workflow} neuron localization",
    )

    plan = plan_from_scores(
        selected, delta=args.neuron_delta, max_scale=args.max_scale
    )
    console.print()
    console.print(
        f"{workflow} plan: {len(plan)} neurons, target "
        f"[bold]{args.target}[/], delta {args.neuron_delta:+.2f}"
    )
    render_plan(console, plan, args.target)
    save_plan_to(console, plan, getattr(args, "save_plan", None))

    evaluation_before: float | None = None
    evaluation_after: float | None = None
    good_before = None
    good_after = None
    if args.evaluate:
        evaluation_before = evaluate_refusal_rate(
            runtime,
            runtime.tokenizer,
            bad_split.evaluation,
            scorer,
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_batch_size,
            max_length=args.max_sequence_length,
        )
        good_before = evaluate_responses(
            runtime,
            runtime.tokenizer,
            good_split.evaluation,
            scorer,
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_batch_size,
            max_length=args.max_sequence_length,
        )
        console.print()
        console.print(
            f"* {metric_label} before edits: {evaluation_before:.3f} "
            f"({evaluation_before * 100:.1f}%)"
        )
        console.print(
            f"* Good-prompt {metric_label.lower()} before edits: "
            f"{good_before.refusal_rate:.3f}, mean length "
            f"{good_before.mean_length:.1f} words"
        )

    if args.dry_run:
        console.print("[yellow]Dry run: no weights were changed.[/]")
        return

    if not args.yes and sys.stdin.isatty():
        import questionary

        confirmed = questionary.confirm(
            f"Apply these {workflow} Tweak edits to the model?", default=True
        ).ask()
        if not confirmed:
            console.print("Edits were not applied.")
            return

    results = runtime.apply_plan(plan, target=args.target)
    console.print()
    render_applied(console, results)

    if args.evaluate:
        evaluation_after = evaluate_refusal_rate(
            runtime,
            runtime.tokenizer,
            bad_split.evaluation,
            scorer,
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_batch_size,
            max_length=args.max_sequence_length,
        )
        good_after = evaluate_responses(
            runtime,
            runtime.tokenizer,
            good_split.evaluation,
            scorer,
            max_new_tokens=args.eval_max_new_tokens,
            batch_size=args.eval_batch_size,
            max_length=args.max_sequence_length,
        )
        console.print()
        console.print(
            f"* {metric_label} after edits: {evaluation_after:.3f} "
            f"({evaluation_after * 100:.1f}%)"
        )
        console.print(
            f"* Good-prompt {metric_label.lower()} after edits: "
            f"{good_after.refusal_rate:.3f}, mean length "
            f"{good_after.mean_length:.1f} words"
        )

    save_directory = args.save_directory
    if save_directory is None and sys.stdin.isatty():
        import questionary

        save_directory = questionary.text(
            "Save directory (leave empty to keep the edit in memory only):"
        ).ask()

    if save_directory:
        saved_path = runtime.save(save_directory)
        manifest_path = write_manifest(
            saved_path,
            build_manifest(runtime.info, args.target, results),
        )
        console.print(f"[green]Saved edited model to {saved_path}[/]")
        console.print(f"* Wrote edit manifest to {manifest_path}")
        console.print(f"* Saved model verified: {runtime.verify_saved(saved_path, results)}")
    else:
        console.print(
            "[yellow]The edit is applied in memory only. No model directory "
            "was written.[/]"
        )
