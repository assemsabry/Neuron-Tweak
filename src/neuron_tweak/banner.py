"""Startup banner for Neuron Tweak.

The banner renders "NEURON TWEAK" as blocky pixel art, in the spirit of the
SHADE banner but in a non-blue color. The color can be overridden with the
NEURON_TWEAK_BANNER_COLOR environment variable (any Rich color name).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rich.console import Console

# Each glyph is five rows tall and exactly five columns wide.
_FONT: dict[str, tuple[str, str, str, str, str]] = {
    "N": (
        "█   █",
        "██  █",
        "█ █ █",
        "█  ██",
        "█   █",
    ),
    "E": (
        "█████",
        "█    ",
        "████ ",
        "█    ",
        "█████",
    ),
    "U": (
        "█   █",
        "█   █",
        "█   █",
        "█   █",
        "█████",
    ),
    "R": (
        "████ ",
        "█   █",
        "████ ",
        "█  █ ",
        "█   █",
    ),
    "O": (
        "█████",
        "█   █",
        "█   █",
        "█   █",
        "█████",
    ),
    "T": (
        "█████",
        "  █  ",
        "  █  ",
        "  █  ",
        "  █  ",
    ),
    "W": (
        "█   █",
        "█   █",
        "█ █ █",
        "██ ██",
        "█   █",
    ),
    "A": (
        "█████",
        "█   █",
        "█████",
        "█   █",
        "█   █",
    ),
    "K": (
        "█   █",
        "█  █ ",
        "███  ",
        "█  █ ",
        "█   █",
    ),
    " ": (
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
    ),
}

_BANNER_TEXT = "NEURON TWEAK"
_DEFAULT_COLOR = "green"


def render_banner_text(text: str = _BANNER_TEXT) -> list[str]:
    """Return the banner as a list of five strings."""
    glyph_rows: list[list[str]] = [[] for _ in range(5)]
    for index, character in enumerate(text.upper()):
        glyph = _FONT[character]
        for row in range(5):
            glyph_rows[row].append(glyph[row])
            if index < len(text) - 1:
                glyph_rows[row].append(" ")
    return ["".join(rows).rstrip() for rows in glyph_rows]


def banner_color() -> str:
    """Resolve the banner color, defaulting to a non-blue terminal color."""
    return os.environ.get("NEURON_TWEAK_BANNER_COLOR", _DEFAULT_COLOR)


def print_banner(console: "Console | None" = None) -> None:
    """Print the Neuron Tweak startup banner."""
    from rich.console import Console

    console = console or Console()
    color = banner_color()
    for line in render_banner_text():
        console.print(f"[bold {color}]{line}[/]")
