"""Command-line entry point for Neuron Tweak."""

from __future__ import annotations

import argparse

from rich.console import Console

from . import __version__
from .banner import print_banner

_PROFILES = {
    "refusal": "Refusal neurons",
    "identity": "Identity neurons",
}


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
        choices=sorted(_PROFILES),
        help="Neuron workflow to run.",
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
            questionary.Choice("Refusal neurons", value="refusal"),
            questionary.Choice("Identity neurons", value="identity"),
        ],
    ).ask()


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

    model = args.model or ask_model(console)
    if not model:
        console.print("[red]No model selected. Exiting.[/]")
        return

    profile = args.profile or ask_profile(console)
    if not profile:
        console.print("[red]No workflow selected. Exiting.[/]")
        return

    console.print()
    console.print(f"* Model: [bold]{model}[/]")
    console.print(f"* Workflow: [bold]{_PROFILES[profile]}[/]")
    console.print("* GPU backend: to be detected during the analysis phase.")
    console.print()
    console.print(
        "[yellow]Neuron analysis and modification are not implemented yet.[/] "
        "This release contains the project skeleton, the startup workflow, and "
        "the built-in prompt datasets (600 good and 600 bad prompts per "
        "workflow)."
    )


if __name__ == "__main__":
    run()
