"""Command-line entry point for Neuron Tweak."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from . import __version__
from .banner import print_banner
from .custom import render_plan, run_custom_pipeline
from .model import ModelRuntime, ModelRuntimeError
from .neurons import NeuronPlan, NeuronSpecError
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
        help="Parse and print the custom plan without loading the model.",
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
        console.print("[bold]Parsed custom neuron plan[/]")
        render_plan(console, plan)
        if not plan:
            console.print(
                "[yellow]No edits were parsed. Pass --custom or --custom-file.[/]"
            )
        return

    console.print()
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
            save_directory=args.save_directory,
            interactive=interactive,
        )
    except ModelRuntimeError as error:
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

    model = args.model
    if model is None and not args.dry_run:
        model = ask_model(console)
    if model is None and not args.dry_run:
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

    console.print(
        f"[yellow]{_PROFILES[profile]} requires the activation-analysis "
        "pipeline, which is not implemented yet.[/] "
        "Custom Tweak is fully implemented; choose it to edit or delete "
        "specific neurons by name."
    )


if __name__ == "__main__":
    run()
