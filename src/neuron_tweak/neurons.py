"""Neuron addressing, steering values, and editable neuron plans.

Custom Tweak uses a compact neuron specification:

    L<layer> N<neuron> <delta>

Examples:

    L14 N8949 +1
    L13 N7956 -1

Conventions:

- ``L`` is 1-based. ``L14`` refers to the 14th transformer block.
- ``N`` is a 0-based index into the MLP intermediate dimension.
- ``delta`` is a signed number. The applied scale is ``1 + delta``, clamped to
  ``[0, max_scale]``. ``+1`` doubles the neuron's contribution, ``-1`` sets it
  to zero (also written ``delete``), and ``-0.5`` halves it.
- ``reset`` (or ``remove`` / ``clear``) removes the entry from the plan and
  restores the original weights when the plan is applied.

The edit targets the matching column of ``down_proj.weight``, whose rows map
the MLP intermediate dimension back to the hidden dimension. Scaling that
column scales the neuron's contribution to the residual stream.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

DEFAULT_MAX_SCALE = 10.0

_RESET_WORDS = {"reset", "remove", "clear"}
_DELETE_WORDS = {"delete", "off", "zero"}

_SPEC_RE = re.compile(
    r"^L\s*(?P<layer>\d+)\s*N\s*(?P<neuron>\d+)"
    r"(?:\s+(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"|reset|remove|clear|delete|off|zero)"
    r"|(?P<signed>[+-](?:\d+(?:\.\d*)?|\.\d+)))$",
    re.IGNORECASE,
)


class NeuronSpecError(ValueError):
    """Raised when a neuron specification is invalid."""


@dataclass(frozen=True)
class ParsedSpec:
    """The result of parsing one neuron specification."""

    layer: int
    neuron: int
    delta: float | None
    original: str


@dataclass(frozen=True)
class NeuronEdit:
    """A single neuron edit: layer, neuron index, and steering delta."""

    layer: int
    neuron: int
    delta: float
    max_scale: float = DEFAULT_MAX_SCALE

    def __post_init__(self) -> None:
        if self.layer < 1:
            raise NeuronSpecError("Layer numbers are 1-based and must be at least 1")
        if self.neuron < 0:
            raise NeuronSpecError("Neuron indices must be zero or greater")
        if not math.isfinite(self.delta):
            raise NeuronSpecError("Steering value must be a finite number")
        if not math.isfinite(self.max_scale) or self.max_scale <= 0:
            raise NeuronSpecError("max_scale must be a positive finite number")

    @property
    def key(self) -> tuple[int, int]:
        return (self.layer, self.neuron)

    @property
    def scale(self) -> float:
        """The multiplicative scale applied to the down_proj column."""
        return max(0.0, min(1.0 + self.delta, self.max_scale))

    @property
    def is_deletion(self) -> bool:
        return self.scale == 0.0

    @property
    def effect(self) -> str:
        scale = self.scale
        if scale == 0.0:
            return "delete"
        if scale < 1.0:
            return f"reduce x{scale:.2f}"
        if scale > 1.0:
            return f"amplify x{scale:.2f}"
        return "no change"

    def to_line(self) -> str:
        return f"L{self.layer} N{self.neuron} {format_delta(self.delta)}"


def format_delta(delta: float) -> str:
    """Format a steering delta as +1, -1, +0.5, and so on."""
    if delta.is_integer():
        return f"{int(delta):+d}"
    text = f"{delta:+.3f}".rstrip("0").rstrip(".")
    return text if text not in {"+", "-"} else f"{delta:+.1f}"


def parse_neuron_spec(spec: str, max_scale: float = DEFAULT_MAX_SCALE) -> ParsedSpec:
    """Parse one L/N/delta specification.

    A delta of None means the specification asks to reset the neuron.
    """
    text = spec.strip()
    if not text:
        raise NeuronSpecError("Empty neuron specification")

    match = _SPEC_RE.match(text)
    if match is None:
        raise NeuronSpecError(
            f"Invalid neuron specification {spec!r}; "
            "expected a form such as 'L14 N8949 +1' or 'L13 N7956 -1'"
        )

    layer = int(match.group("layer"))
    neuron = int(match.group("neuron"))
    value = (match.group("value") or match.group("signed") or "").lower()

    if layer < 1:
        raise NeuronSpecError("Layer numbers are 1-based and must be at least 1")
    if neuron < 0:
        raise NeuronSpecError("Neuron indices must be zero or greater")

    if value in _RESET_WORDS:
        return ParsedSpec(layer=layer, neuron=neuron, delta=None, original=text)

    if value in _DELETE_WORDS:
        delta = -1.0
    else:
        delta = float(value)

    if not math.isfinite(delta):
        raise NeuronSpecError("Steering value must be a finite number")

    return ParsedSpec(layer=layer, neuron=neuron, delta=delta, original=text)


@dataclass
class NeuronPlan:
    """An ordered, editable set of neuron edits keyed by (layer, neuron)."""

    max_scale: float = DEFAULT_MAX_SCALE
    edits: dict[tuple[int, int], NeuronEdit] = field(default_factory=dict)

    def add(self, edit: NeuronEdit) -> None:
        if edit.max_scale != self.max_scale:
            edit = NeuronEdit(
                layer=edit.layer,
                neuron=edit.neuron,
                delta=edit.delta,
                max_scale=self.max_scale,
            )
        self.edits[edit.key] = edit

    def remove(self, layer: int, neuron: int) -> bool:
        return self.edits.pop((layer, neuron), None) is not None

    def get(self, layer: int, neuron: int) -> NeuronEdit | None:
        return self.edits.get((layer, neuron))

    def clear(self) -> None:
        self.edits.clear()

    def sorted(self) -> list[NeuronEdit]:
        return [self.edits[key] for key in sorted(self.edits)]

    def add_specs(self, text: str) -> list[ParsedSpec]:
        """Add one or more specifications separated by commas, semicolons, or newlines."""
        tokens = [token.strip() for token in re.split(r"[,;\n]+", text) if token.strip()]
        if not tokens:
            raise NeuronSpecError("No neuron specifications were provided")

        parsed_specs = [parse_neuron_spec(token, self.max_scale) for token in tokens]
        for parsed in parsed_specs:
            if parsed.delta is None:
                self.remove(parsed.layer, parsed.neuron)
            else:
                self.add(
                    NeuronEdit(
                        layer=parsed.layer,
                        neuron=parsed.neuron,
                        delta=parsed.delta,
                        max_scale=self.max_scale,
                    )
                )
        return parsed_specs

    def to_text(self) -> str:
        return "\n".join(edit.to_line() for edit in self.sorted())

    @classmethod
    def from_text(cls, text: str, max_scale: float = DEFAULT_MAX_SCALE) -> "NeuronPlan":
        plan = cls(max_scale=max_scale)
        plan.add_specs(text)
        return plan

    def __len__(self) -> int:
        return len(self.edits)

    def __iter__(self):
        return iter(self.sorted())

    def __bool__(self) -> bool:
        return bool(self.edits)
