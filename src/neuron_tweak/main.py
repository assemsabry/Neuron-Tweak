"""Command-line entry point for Neuron Tweak."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from . import __version__
from .analysis import AnalysisError
from .banner import print_banner
from .custom import render_plan, run_custom_pipeline
from .model import TARGETS, ModelRuntime, ModelRuntimeError
from .neurons import NeuronPlan, NeuronSpecError
from .safety import run_safety_pipeline
from .system import detect_backend

_PROFILES = {
    "safety": "Safety Tweak",
    "identity": "Identity Tweak",
    "custom": "Custom Tweak",
}

_PROFILE_ALIASES = {
    "refusal": "safety",
}


def normalize_profile(value: str) -> str:
    key = value.strip().lower()
    return _PROFILE_ALIASES.get(key, key)


def profile_type(value: str) -> str:
    key = normalize_profile(value)
    if key not in _PROFILES:
        choices = ", ".join(sorted(_PROFILES))
        raise argparse.ArgumentTypeError(
            f"unknown workflow {value!r}; choose one of: {choices}"
        )
    return key


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuron-tweak",
        description=(
            "Training-free neuron-level modification for local language models."
        ),
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="Model ID or local path to load.",
    )
    parser.add_argument(
        "--profile",
        type=profile_type,
        metavar="{safety,identity,custom}",
        help="Neuron workflow to run.",
    )
    parser.add_argument(
        "--custom",
        action="append",
        metavar="SPEC",
        help=(
            "Custom neuron edit such as 'L14 N8949 +1'. Can be repeated or "
            "contain comma-separated entries."
        ),
    )
    parser.add_argument(
        "--custom-file",
        metavar="PATH",
        help="Text file with one or more custom neuron specifications.",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=10.0,
        help="Maximum multiplicative scale for a custom edit (default: 10).",
    )
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="down_proj",
        help=(
            "MLP projection to edit: down_proj (default), up_proj, gate_proj, "
            "or all."
        ),
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=100,
        help="Held-out prompts per class for Safety/Identity (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for the prompt split (default: 0).",
    )
    parser.add_argument(
        "--analysis-batch-size",
        type=int,
        default=8,
        help="Batch size for activation capture (default: 8).",
    )
    parser.add_argument(
        "--analysis-prompts",
        type=int,
        default=0,
        help=(
            "Limit training prompts per class for a quick analysis "
            "(default: 0, use every training prompt)."
        ),
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=512,
        help="Maximum tokens per analysis prompt (default: 512).",
    )
    parser.add_argument(
        "--top-neurons",
        type=int,
        default=10,
        help="Neurons to select per layer for Safety/Identity (default: 10).",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum Cohen's d for a selected neuron (default: 0).",
    )
    parser.add_argument(
        "--neuron-delta",
        type=float,
        default=-1.0,
        help="Delta applied to selected Safety/Identity neurons (default: -1).",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Layers to analyze, for example 'all', '10-20', or '14,15'.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Apply Safety/Identity edits without a confirmation prompt.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Generate before/after responses and report the refusal rate.",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for evaluation responses (default: 64).",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation responses (default: 4).",
    )
    parser.add_argument(
        "--study",
        action="store_true",
        help="Run an Optuna study over the selected safety neurons.",
    )
    parser.add_argument(
        "--study-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for the safety study (default: 30).",
    )
    parser.add_argument(
        "--study-startup-trials",
        type=int,
        default=10,
        help="Startup trials for the Optuna safety study (default: 10).",
    )
    parser.add_argument(
        "--study-direction",
        choices=["minimize", "maximize"],
        default="minimize",
        help="Study direction for the safety Optuna study (default: minimize).",
    )
    parser.add_argument(
        "--study-apply-best",
        action="store_true",
        help="Apply the best safety study trial and save it.",
    )
    parser.add_argument(
        "--save-directory",
        "-o",
        metavar="PATH",
        help="Directory where the edited model is saved.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Open the Custom Tweak editor even when specifications are provided.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Custom: parse and print the plan without loading the model. "
            "Safety/Identity: analyze and print the plan without changing weights."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Model dtype (default: auto).",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Accelerate device map (default: auto).",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "bnb_4bit"],
        default="none",
        help="Quantization mode. Custom neuron editing requires none.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model code from the model repository.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def ask_model(console: Console) -> str | None:
    import questionary

    console.print()
    console.print("[bold]How would you like to load a model?[/]")
    return questionary.text(
        "Enter a Hugging Face model ID or a local model path:",
        validate=lambda value: bool(value.strip()) or "A model ID or path is required.",
    ).ask()


def ask_profile(console: Console) -> str | None:
    import questionary

    console.print()
    console.print("[bold]Which neurons do you want to modify?[/]")
    return questionary.select(
        "Select a workflow:",
        choices=[
            questionary.Choice("Safety Tweak", value="safety"),
            questionary.Choice("Identity Tweak", value="identity"),
            questionary.Choice("Custom Tweak", value="custom"),
        ],
    ).ask()


def collect_custom_specs(args: argparse.Namespace) -> list[str]:
    specs = list(args.custom or [])
    if args.custom_file:
        path = Path(args.custom_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Custom specification file not found: {path}")
        specs.append(path.read_text(encoding="utf-8"))
    return specs


def run_custom(args: argparse.Namespace, model: str, console: Console) -> None:
    if args.max_scale <= 0:
        console.print("[red]--max-scale must be greater than zero.[/]")
        return

    plan = NeuronPlan(max_scale=args.max_scale)
    try:
        specs = collect_custom_specs(args)
        for spec in specs:
            plan.add_specs(spec)
    except (NeuronSpecError, OSError) as error:
        console.print(f"[red]{error}[/]")
        return

    if args.dry_run:
        console.print()
        console.print(
            f"[bold]Parsed custom neuron plan[/] (target: {args.target})"
        )
        render_plan(console, plan, args.target)
        if not plan:
            console.print(
                "[yellow]No edits were parsed. Pass --custom or --custom-file.[/]"
            )
        return

    console.print()
    console.print(f"Target projection: [bold]{args.target}[/]")
    console.print("[bold]Loading model...[/]")
    runtime = ModelRuntime(
        model,
        dtype=args.dtype,
        device_map=args.device_map,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        info = runtime.load()
        console.print()
        for line in info.lines():
            console.print(f"* {line}")

        interactive = args.interactive or not specs
        run_custom_pipeline(
            runtime,
            plan,
            console,
            target=args.target,
            save_directory=args.save_directory,
            interactive=interactive,
        )
    except ModelRuntimeError as error:
        console.print(f"[red]{error}[/]")
    finally:
        runtime.close()


def run_safety(args: argparse.Namespace, model: str, console: Console) -> None:
    for flag, value in (
        ("--max-scale", args.max_scale),
        ("--top-neurons", args.top_neurons),
        ("--analysis-batch-size", args.analysis_batch_size),
        ("--max-sequence-length", args.max_sequence_length),
        ("--evaluation-size", args.evaluation_size),
    ):
        if value <= 0:
            console.print(f"[red]{flag} must be greater than zero.[/]")
            return
    if args.analysis_prompts < 0:
        console.print("[red]--analysis-prompts cannot be negative.[/]")
        return
    if getattr(args, "evaluate", False):
        for flag, value in (
            ("--eval-max-new-tokens", args.eval_max_new_tokens),
            ("--eval-batch-size", args.eval_batch_size),
        ):
            if value <= 0:
                console.print(f"[red]{flag} must be greater than zero.[/]")
                return
    if getattr(args, "study", False):
        if args.study_trials <= 0:
            console.print("[red]--study-trials must be greater than zero.[/]")
            return
        if args.study_startup_trials < 0:
            console.print(
                "[red]--study-startup-trials cannot be negative.[/]"
            )
            return

    console.print("[bold]Loading model...[/]")
    runtime = ModelRuntime(
        model,
        dtype=args.dtype,
        device_map=args.device_map,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
    )
    try:
        info = runtime.load()
        console.print()
        for line in info.lines():
            console.print(f"* {line}")
        console.print()
        run_safety_pipeline(runtime, console, args)
    except (ModelRuntimeError, AnalysisError, NeuronSpecError) as error:
        console.print(f"[red]{error}[/]")
    finally:
        runtime.close()


def run() -> None:
    parser = build_parser()
    args = parser.parse_args()

    console = Console()
    print_banner(console)
    console.print(f"[bold]Neuron Tweak[/] v{__version__}")
    console.print(
        "Training-free neuron-level modification for local language models."
    )
    console.print()

    custom_dry_run = args.profile == "custom" and args.dry_run
    model = args.model
    if model is None and not custom_dry_run:
        model = ask_model(console)
    if model is None and not custom_dry_run:
        console.print("[red]No model selected. Exiting.[/]")
        return

    profile = args.profile or ask_profile(console)
    if not profile:
        console.print("[red]No workflow selected. Exiting.[/]")
        return

    backend = detect_backend()
    console.print()
    console.print(f"* Model: [bold]{model or '(not loaded)'}[/]")
    console.print(f"* Workflow: [bold]{_PROFILES[profile]}[/]")
    console.print(f"* GPU backend: [bold]{backend.summary()}[/]")
    if backend.backend == "CPU":
        console.print(
            "[yellow]No GPU backend was detected. Neuron editing can run on the "
            "CPU, but it will be slow and memory-hungry.[/]"
        )
    console.print()

    if profile == "custom":
        run_custom(args, model or "", console)
        return
    if profile == "safety":
        run_safety(args, model or "", console)
        return

    console.print(
        "[yellow]Identity Tweak requires the identity activation pipeline, "
        "which is not implemented yet.[/] Safety Tweak and Custom Tweak are "
        "available now."
    )


if __name__ == "__main__":
    run()
