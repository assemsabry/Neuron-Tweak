"""Custom Tweak: user-directed MLP neuron edits.

The editor lets the user add, update, and remove entries in a ``NeuronPlan``
using specifications such as ``L14 N8949 +1`` and ``L13 N7956 -1``. The plan is
then applied to the model's ``down_proj`` columns and can be saved as a normal
Hugging Face model directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from rich.table import Table

from .evaluator import evaluate_refusal_rate, evaluate_responses
from .manifest import build_manifest, write_manifest
from .neurons import NeuronPlan, NeuronSpecError, format_delta
from .prompts import PromptDatasetError, load_workflow_prompts
from .scorer import RefusalRateScorer

if TYPE_CHECKING:
    from rich.console import Console

    from .model import AppliedEdit, ModelRuntime


def save_plan_to(
    console: "Console", plan: NeuronPlan, path: str | None
) -> bool:
    """Write the plan as text and report the path; return whether it saved."""
    if not path or not plan:
        return False
    try:
        saved = plan.save(path)
    except OSError as error:
        console.print(f"[red]Could not write the plan: {error}[/]")
        return False
    console.print(f"* Wrote neuron plan to {saved}")
    return True


def render_plan(
    console: "Console", plan: NeuronPlan, target: str = "down_proj"
) -> None:
    """Print the current custom edit plan as a table."""
    if not plan:
        console.print("[dim]The neuron edit plan is empty.[/]")
        return

    table = Table(title=f"Neuron edit plan (target: {target})", show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Neuron", style="cyan")
    table.add_column("Delta", justify="right")
    table.add_column("Scale", justify="right")
    table.add_column("Effect")

    for index, edit in enumerate(plan, start=1):
        style = "red" if edit.is_deletion else "green" if edit.scale > 1 else "yellow"
        table.add_row(
            str(index),
            f"L{edit.layer} N{edit.neuron}",
            format_delta(edit.delta),
            f"x{edit.scale:.2f}",
            edit.effect,
            style=style,
        )
    console.print(table)


def render_applied(console: "Console", results: list["AppliedEdit"]) -> None:
    """Print the edits that were applied to the model."""
    table = Table(title="Applied neuron edits")
    table.add_column("Neuron", style="cyan")
    table.add_column("Projection")
    table.add_column("Scale", justify="right")
    table.add_column("Effect")
    table.add_column("Vector norm", justify="right")

    for result in results:
        edit = result.edit
        table.add_row(
            f"L{edit.layer} N{edit.neuron}",
            result.projection,
            f"x{edit.scale:.2f}",
            edit.effect,
            f"{result.original_norm:.4f} -> {result.new_norm:.4f}",
        )
    console.print(table)


def edit_plan_interactive(
    console: "Console", plan: NeuronPlan, target: str = "down_proj"
) -> None:
    """Run the add/remove/clear editor loop until the user chooses Done."""
    import questionary

    while True:
        console.print()
        render_plan(console, plan, target)
        action = questionary.select(
            "Custom Tweak editor",
            choices=[
                "Add or update a neuron edit",
                "Remove an entry from the plan",
                "Clear the plan",
                "Show syntax help",
                "Done",
            ],
        ).ask()

        if action is None or action == "Done":
            return

        if action == "Add or update a neuron edit":
            spec = questionary.text(
                "Neuron spec (examples: L14 N8949 +1, L13 N7956 -1):"
            ).ask()
            if not spec:
                continue
            try:
                parsed = plan.add_specs(spec)
            except NeuronSpecError as error:
                console.print(f"[red]{error}[/]")
                continue
            for entry in parsed:
                if entry.delta is None:
                    console.print(f"Removed L{entry.layer} N{entry.neuron}")
                else:
                    console.print(
                        f"Saved L{entry.layer} N{entry.neuron} "
                        f"{format_delta(entry.delta)}"
                    )
            continue

        if action == "Remove an entry from the plan":
            entries = plan.sorted()
            if not entries:
                console.print("[yellow]The plan is empty.[/]")
                continue
            selected = questionary.select(
                "Remove which entry?",
                choices=[
                    questionary.Choice(
                        title=(
                            f"L{edit.layer} N{edit.neuron} "
                            f"({format_delta(edit.delta)}, {edit.effect})"
                        ),
                        value=edit,
                    )
                    for edit in entries
                ]
                + ["Cancel"],
            ).ask()
            if selected in (None, "Cancel"):
                continue
            plan.remove(selected.layer, selected.neuron)
            console.print(f"Removed L{selected.layer} N{selected.neuron}")
            continue

        if action == "Clear the plan":
            if questionary.confirm("Clear every entry in the plan?").ask():
                plan.clear()
                console.print("Plan cleared.")
            continue

        if action == "Show syntax help":
            console.print(
                "[bold]Syntax[/]\n"
                "  L<layer> N<neuron> <delta>\n\n"
                "  L is 1-based. N is a 0-based index into the MLP "
                "intermediate dimension.\n"
                "  +1 doubles the neuron, -1 deletes it, -0.5 halves it.\n"
                "  delete is an alias for -1. reset removes the plan entry.\n"
                "  Multiple entries may be separated by commas or semicolons.\n"
                f"  Current target projection: {target}.\n"
            )


def resolve_evaluation_prompts(
    console: "Console",
    good_prompts: Sequence[str] | None,
    bad_prompts: Sequence[str] | None,
    evaluation_size: int,
    seed: int | None,
) -> tuple[list[str], list[str]]:
    """Use injected evaluation prompts, or load the safety held-out sets."""
    good = list(good_prompts or [])
    bad = list(bad_prompts or [])
    if not good or not bad:
        try:
            good_split, bad_split = load_workflow_prompts(
                "safety",
                evaluation_size=evaluation_size,
                seed=seed,
            )
        except PromptDatasetError as error:
            console.print(f"[red]{error}[/]")
            return ([], [])
        good = list(good_split.evaluation)
        bad = list(bad_split.evaluation)
    return good, bad


def run_custom_pipeline(
    runtime: "ModelRuntime",
    plan: NeuronPlan,
    console: "Console",
    *,
    target: str = "down_proj",
    save_directory: str | None = None,
    save_plan: str | None = None,
    evaluate: bool = False,
    good_prompts: Sequence[str] | None = None,
    bad_prompts: Sequence[str] | None = None,
    evaluation_size: int = 100,
    seed: int | None = 0,
    eval_max_new_tokens: int = 64,
    eval_batch_size: int = 4,
    max_sequence_length: int = 512,
    interactive: bool = True,
) -> None:
    """Edit the plan, apply it to the model, and optionally save the result."""
    if interactive:
        edit_plan_interactive(console, plan, target)

    if not plan:
        console.print("[yellow]No neuron edits to apply. Exiting.[/]")
        return

    console.print()
    console.print(f"Target projection: [bold]{target}[/]")
    render_plan(console, plan, target)
    save_plan_to(console, plan, save_plan)

    evaluation_before: float | None = None
    evaluation_after: float | None = None
    good_before = None
    good_after = None
    if evaluate:
        good_prompts, bad_prompts = resolve_evaluation_prompts(
            console,
            good_prompts,
            bad_prompts,
            evaluation_size,
            seed,
        )
        if good_prompts and bad_prompts:
            scorer = RefusalRateScorer()
            evaluation_before = evaluate_refusal_rate(
                runtime,
                runtime.tokenizer,
                bad_prompts,
                scorer,
                max_new_tokens=eval_max_new_tokens,
                batch_size=eval_batch_size,
                max_length=max_sequence_length,
            )
            good_before = evaluate_responses(
                runtime,
                runtime.tokenizer,
                good_prompts,
                scorer,
                max_new_tokens=eval_max_new_tokens,
                batch_size=eval_batch_size,
                max_length=max_sequence_length,
            )
            console.print()
            console.print(
                f"* Refusal rate before edits: {evaluation_before:.3f} "
                f"({evaluation_before * 100:.1f}%)"
            )
            console.print(
                f"* Good-prompt metrics before edits: refusal rate "
                f"{good_before.refusal_rate:.3f}, mean length "
                f"{good_before.mean_length:.1f} words"
            )

    if interactive:
        import questionary

        confirmed = questionary.confirm(
            "Apply these edits to the model now?", default=True
        ).ask()
        if not confirmed:
            console.print("Edits were not applied.")
            return

    try:
        results = runtime.apply_plan(plan, target=target)
    except Exception as error:  # noqa: BLE001 - show a friendly editor error
        console.print(f"[red]Could not apply the plan: {error}[/]")
        return

    render_applied(console, results)

    if evaluate and evaluation_before is not None and good_before is not None:
        scorer = RefusalRateScorer()
        evaluation_after = evaluate_refusal_rate(
            runtime,
            runtime.tokenizer,
            bad_prompts,
            scorer,
            max_new_tokens=eval_max_new_tokens,
            batch_size=eval_batch_size,
            max_length=max_sequence_length,
        )
        good_after = evaluate_responses(
            runtime,
            runtime.tokenizer,
            good_prompts,
            scorer,
            max_new_tokens=eval_max_new_tokens,
            batch_size=eval_batch_size,
            max_length=max_sequence_length,
        )
        console.print()
        console.print(
            f"* Refusal rate after edits: {evaluation_after:.3f} "
            f"({evaluation_after * 100:.1f}%)"
        )
        console.print(
            f"* Good-prompt metrics after edits: refusal rate "
            f"{good_after.refusal_rate:.3f}, mean length "
            f"{good_after.mean_length:.1f} words"
        )

    if save_directory is None and interactive:
        import questionary

        save_directory = questionary.text(
            "Save directory (leave empty to keep the edit in memory only):"
        ).ask()

    if save_directory:
        try:
            saved_path = runtime.save(save_directory)
        except Exception as error:  # noqa: BLE001 - show a friendly save error
            console.print(f"[red]Could not save the model: {error}[/]")
            return
        manifest_path = write_manifest(
            saved_path, build_manifest(runtime.info, target, results)
        )
        console.print(f"[green]Saved edited model to {saved_path}[/]")
        console.print(f"* Wrote edit manifest to {manifest_path}")
    else:
        console.print(
            "[yellow]The edit is applied in memory only. No model directory was "
            "written.[/]"
        )
