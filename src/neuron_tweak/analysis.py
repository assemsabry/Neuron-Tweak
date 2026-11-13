"""Activation capture and neuron scoring for Safety and Identity Tweak.

The analysis pipeline runs contrastive prompts through the model and records
the input activations of each MLP down-projection. Each column of that input is
one MLP neuron. Comparing the mean activation on "bad" prompts with the mean on
"good" prompts gives a per-neuron score. Neurons with a high positive Cohen's d
are more active on bad prompts and are candidates for suppression.

The module is import-safe without PyTorch. Torch is only imported when a
capture or scoring operation actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .neurons import NeuronEdit, NeuronPlan

if TYPE_CHECKING:
    from .model import ModelRuntime

ProgressCallback = Callable[[int, int], None]


class AnalysisError(RuntimeError):
    """Raised when activation capture or scoring cannot proceed."""


@dataclass(frozen=True)
class NeuronScore:
    """Contrastive activation score for one MLP neuron."""

    layer: int  # 1-based transformer block
    neuron: int  # 0-based MLP intermediate index
    mean_good: float
    mean_bad: float
    cohen_d: float
    mean_diff: float

    @property
    def address(self) -> str:
        return f"L{self.layer} N{self.neuron}"


@dataclass
class ActivationStats:
    """Accumulated per-neuron activation statistics for one prompt class."""

    sums: list[Any]
    sums_sq: list[Any]
    counts: list[int]
    intermediate_size: int
    label: str = ""

    @property
    def num_layers(self) -> int:
        return len(self.sums)

    def means(self, layer_index: int) -> Any:
        count = self.counts[layer_index]
        if count <= 0:
            raise AnalysisError(
                f"No activations were captured for layer index {layer_index}"
            )
        return self.sums[layer_index] / count

    def variances(self, layer_index: int) -> Any:
        mean = self.means(layer_index)
        return self.sums_sq[layer_index] / self.counts[layer_index] - mean * mean


class ActivationCapture:
    """Capture down-projection inputs from every selected transformer layer.

    Use as a context manager:

        with ActivationCapture(runtime) as capture:
            runtime.model(input_ids=...)
        stats = capture.stats
    """

    def __init__(
        self,
        runtime: "ModelRuntime",
        *,
        label: str = "",
        layer_numbers: Sequence[int] | None = None,
    ) -> None:
        if not runtime.is_loaded:
            raise AnalysisError("Model is not loaded")
        try:
            import torch
        except ModuleNotFoundError as error:  # pragma: no cover - runtime guard
            raise AnalysisError("PyTorch is not installed") from error

        self.runtime = runtime
        self.torch = torch
        self.label = label
        self.intermediate_size = runtime.info.intermediate_size
        self.layer_numbers = list(
            layer_numbers
            if layer_numbers is not None
            else range(1, runtime.info.num_layers + 1)
        )
        if not self.layer_numbers:
            raise AnalysisError("At least one layer must be selected")

        self.stats = self._empty_stats()
        self._handles: list[Any] = []

    def _empty_stats(self) -> ActivationStats:
        size = self.intermediate_size
        return ActivationStats(
            sums=[
                self.torch.zeros(size, dtype=self.torch.float32)
                for _ in self.layer_numbers
            ],
            sums_sq=[
                self.torch.zeros(size, dtype=self.torch.float32)
                for _ in self.layer_numbers
            ],
            counts=[0 for _ in self.layer_numbers],
            intermediate_size=size,
            label=self.label,
        )

    def reset(self) -> None:
        """Clear all accumulated activations."""
        for index in range(len(self.layer_numbers)):
            self.stats.sums[index].zero_()
            self.stats.sums_sq[index].zero_()
            self.stats.counts[index] = 0

    def __enter__(self) -> "ActivationCapture":
        if self._handles:
            return self
        for layer_number in self.layer_numbers:
            module = self.runtime.projection_module(layer_number, "down_proj")
            handle = module.register_forward_pre_hook(
                self._make_hook(layer_number)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        """Remove all hooks. Safe to call more than once."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, layer_number: int) -> Callable[..., None]:
        index = self.layer_numbers.index(layer_number)

        def hook(module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or inputs[0] is None:
                return
            tensor = inputs[0]
            if tensor.ndim < 2:
                raise AnalysisError(
                    f"Layer {layer_number} down-projection input has shape "
                    f"{tuple(tensor.shape)}, expected at least 2 dimensions"
                )
            if tensor.shape[-1] != self.intermediate_size:
                raise AnalysisError(
                    f"Layer {layer_number} down-projection input has "
                    f"{tensor.shape[-1]} neurons, expected "
                    f"{self.intermediate_size}"
                )
            flat = tensor.detach().reshape(-1, tensor.shape[-1]).float()
            self.stats.sums[index].add_(flat.sum(dim=0).cpu())
            self.stats.sums_sq[index].add_((flat * flat).sum(dim=0).cpu())
            self.stats.counts[index] += int(flat.shape[0])

        return hook


def _model_device(runtime: "ModelRuntime") -> Any:
    try:
        return runtime.model.device
    except AttributeError:
        return next(runtime.model.parameters()).device


def _format_prompt(
    tokenizer: Any,
    prompt: str,
    system_prompt: str | None = None,
) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:  # noqa: BLE001 - fall back to the raw prompt
        return prompt


def capture_activations(
    runtime: "ModelRuntime",
    tokenizer: Any | None,
    prompts: Sequence[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
    layer_numbers: Sequence[int] | None = None,
    label: str = "",
    system_prompt: str | None = None,
    progress: ProgressCallback | None = None,
) -> ActivationStats:
    """Run prompts through the model and return per-neuron statistics."""
    if not runtime.is_loaded:
        raise AnalysisError("Model is not loaded")
    if tokenizer is None:
        raise AnalysisError(
            "A tokenizer is required for activation capture. This model was "
            "loaded without a tokenizer."
        )
    if batch_size <= 0:
        raise AnalysisError("batch_size must be greater than zero")
    if not prompts:
        raise AnalysisError("No prompts were provided")

    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - runtime guard
        raise AnalysisError("PyTorch is not installed") from error

    if getattr(tokenizer, "pad_token", None) is None:
        eos = getattr(tokenizer, "eos_token", None)
        if eos is None:
            raise AnalysisError(
                "The tokenizer has no pad token or EOS token; cannot batch "
                "prompts."
            )
        tokenizer.pad_token = eos

    device = _model_device(runtime)
    total = len(prompts)
    capture = ActivationCapture(
        runtime, label=label, layer_numbers=layer_numbers
    )

    with torch.inference_mode(), capture:
        for start in range(0, total, batch_size):
            batch = list(prompts[start : start + batch_size])
            texts = [
                _format_prompt(tokenizer, prompt, system_prompt=system_prompt)
                for prompt in batch
            ]
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in encoded.items()
            }
            try:
                runtime.model(**encoded, use_cache=False)
            except TypeError:
                runtime.model(**encoded)
            if progress is not None:
                progress(min(start + len(batch), total), total)

    return capture.stats


def score_neurons(
    good: ActivationStats,
    bad: ActivationStats,
    *,
    eps: float = 1e-6,
) -> list[list[NeuronScore]]:
    """Compute Cohen's d for every neuron, grouped by layer.

    Positive scores mean the neuron is more active on bad prompts. The score is
    ``(mean_bad - mean_good) / pooled_std``.
    """
    if good.num_layers != bad.num_layers:
        raise AnalysisError("Good and bad activation sets have different layers")
    if good.intermediate_size != bad.intermediate_size:
        raise AnalysisError(
            "Good and bad activation sets have different intermediate sizes"
        )
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - runtime guard
        raise AnalysisError("PyTorch is not installed") from error

    scores_by_layer: list[list[NeuronScore]] = []
    for layer_index in range(good.num_layers):
        if good.counts[layer_index] <= 0 or bad.counts[layer_index] <= 0:
            raise AnalysisError(
                f"No activations were captured for layer index {layer_index}"
            )

        good_mean = good.means(layer_index)
        bad_mean = bad.means(layer_index)
        good_var = good.variances(layer_index)
        bad_var = bad.variances(layer_index)
        pooled = torch.sqrt(
            torch.clamp((good_var + bad_var) / 2.0, min=0.0) + eps
        )
        cohen_d = (bad_mean - good_mean) / pooled
        mean_diff = bad_mean - good_mean

        layer_scores = [
            NeuronScore(
                layer=layer_index + 1,
                neuron=neuron,
                mean_good=float(good_mean[neuron].item()),
                mean_bad=float(bad_mean[neuron].item()),
                cohen_d=float(cohen_d[neuron].item()),
                mean_diff=float(mean_diff[neuron].item()),
            )
            for neuron in range(good.intermediate_size)
        ]
        scores_by_layer.append(layer_scores)

    return scores_by_layer


def top_neurons(
    scores_by_layer: Sequence[Sequence[NeuronScore]],
    *,
    per_layer: int = 10,
    layers: Sequence[int] | None = None,
    min_score: float = 0.0,
) -> list[NeuronScore]:
    """Return the highest-scoring neurons from each selected layer."""
    if per_layer <= 0:
        raise AnalysisError("per_layer must be greater than zero")
    selected_layers = set(layers) if layers is not None else None
    selected: list[NeuronScore] = []

    for layer_scores in scores_by_layer:
        if not layer_scores:
            continue
        layer_number = layer_scores[0].layer
        if selected_layers is not None and layer_number not in selected_layers:
            continue
        candidates = [
            score for score in layer_scores if score.cohen_d >= min_score
        ]
        candidates.sort(key=lambda score: score.cohen_d, reverse=True)
        selected.extend(candidates[:per_layer])
    return selected


def global_top_neurons(
    scores_by_layer: Sequence[Sequence[NeuronScore]],
    *,
    count: int = 50,
    min_score: float = 0.0,
) -> list[NeuronScore]:
    """Return the highest-scoring neurons across the whole model."""
    if count <= 0:
        raise AnalysisError("count must be greater than zero")
    candidates = [
        score
        for layer_scores in scores_by_layer
        for score in layer_scores
        if score.cohen_d >= min_score
    ]
    candidates.sort(key=lambda score: score.cohen_d, reverse=True)
    return candidates[:count]


def plan_from_scores(
    scores: Sequence[NeuronScore],
    *,
    delta: float = -1.0,
    max_scale: float = 10.0,
) -> NeuronPlan:
    """Build a Custom Tweak plan that suppresses the selected neurons."""
    plan = NeuronPlan(max_scale=max_scale)
    for score in scores:
        plan.add(NeuronEdit(layer=score.layer, neuron=score.neuron, delta=delta))
    return plan
