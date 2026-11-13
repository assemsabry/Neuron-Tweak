"""Model loading, MLP neuron editing, and saving.

Custom Tweak edits one or more MLP projections of selected neurons. For a
Llama-style block:

    up_proj.weight:    (intermediate_size, hidden_size)
    gate_proj.weight:  (intermediate_size, hidden_size)
    down_proj.weight:  (hidden_size, intermediate_size)

Rows of ``up_proj`` and ``gate_proj`` correspond to MLP neurons. Column ``i``
of ``down_proj`` maps neuron ``i`` back to the hidden dimension. Scaling the
matching rows and column changes the neuron's contribution to the residual
stream. A scale of zero removes the neuron's contribution.

The default target is ``down_proj`` because its column is the neuron's output
path. ``up_proj``, ``gate_proj``, and ``all`` are also available for full
control over the MLP.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .neurons import NeuronEdit, NeuronPlan

PROJECTIONS = ("up_proj", "gate_proj", "down_proj")
TARGETS = ("down_proj", "up_proj", "gate_proj", "all")


class ModelRuntimeError(RuntimeError):
    """Raised when a model cannot be loaded, inspected, or edited."""


@dataclass(frozen=True)
class ModelInfo:
    """A concise description of the loaded model."""

    model_id: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    dtype: str
    device: str

    def lines(self) -> list[str]:
        return [
            f"Model: {self.model_id}",
            f"Layers: {self.num_layers}",
            f"Hidden size: {self.hidden_size}",
            f"MLP intermediate size: {self.intermediate_size}",
            f"Dtype: {self.dtype}",
            f"Device: {self.device}",
        ]


@dataclass(frozen=True)
class AppliedEdit:
    """The result of applying one neuron edit to one projection."""

    edit: NeuronEdit
    projection: str
    original_norm: float
    new_norm: float


class ModelRuntime:
    """Load a local causal LM and edit its MLP neurons on the selected device."""

    def __init__(
        self,
        model_id: str,
        *,
        dtype: str = "auto",
        device_map: str = "auto",
        quantization: str = "none",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.dtype = dtype
        self.device_map = device_map
        self.quantization = quantization
        self.trust_remote_code = trust_remote_code

        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self._layers: list[Any] = []
        self._originals: dict[tuple[str, int, int], Any] = {}
        self._torch: Any | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> ModelInfo:
        """Load the model and tokenizer, trying dtypes in order."""
        if self.is_loaded:
            return self.info

        torch = self._import_torch()
        transformers = self._import_transformers()
        self._torch = torch

        if self.quantization not in {"", "none"}:
            raise ModelRuntimeError(
                "Custom neuron editing requires full-precision weights. "
                f"Quantization {self.quantization!r} is not supported for this "
                "workflow. Load the model in float16 or bfloat16 instead."
            )

        dtype_candidates = self._dtype_candidates()
        dtype_key = self._dtype_kwarg_name(transformers)
        errors: list[str] = []
        model = None
        used_dtype = None

        for dtype_name in dtype_candidates:
            kwargs = {
                "device_map": self.device_map,
                "trust_remote_code": self.trust_remote_code,
                "low_cpu_mem_usage": True,
            }
            dtype_value = self._torch_dtype(torch, dtype_name)
            if dtype_value != "auto":
                kwargs[dtype_key] = dtype_value
            try:
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    self.model_id, **kwargs
                )
                used_dtype = dtype_name
                break
            except Exception as error:  # noqa: BLE001 - report all dtype failures
                errors.append(f"{dtype_name}: {error}")

        if model is None:
            details = "\n".join(f"- {error}" for error in errors)
            raise ModelRuntimeError(
                f"Could not load model {self.model_id!r}:\n{details}"
            )

        model.eval()
        model.requires_grad_(False)
        self.model = model
        self.dtype = used_dtype or self.dtype

        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=self.trust_remote_code
            )
        except Exception:  # noqa: BLE001 - editing can still work without it
            self.tokenizer = None

        self._layers = self._resolve_layers()
        if not self._layers:
            raise ModelRuntimeError(
                "Could not find transformer layers in this model. "
                "Custom Tweak currently supports Llama/Qwen/Mistral-style "
                "models with model.model.layers and mlp.down_proj."
            )
        return self.info

    @property
    def info(self) -> ModelInfo:
        if not self.is_loaded:
            raise ModelRuntimeError("Model is not loaded")

        first_weight = self.down_proj_weight(1)
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        if not hidden_size:
            hidden_size = int(first_weight.shape[0])
        intermediate_size = int(
            getattr(self.model.config, "intermediate_size", 0) or 0
        )
        if not intermediate_size:
            intermediate_size = int(first_weight.shape[1])

        parameter = next(self.model.parameters())
        return ModelInfo(
            model_id=self.model_id,
            num_layers=len(self._layers),
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=str(parameter.dtype).replace("torch.", ""),
            device=str(parameter.device),
        )

    def validate_plan(self, plan: NeuronPlan, target: str = "down_proj") -> None:
        """Raise a single error listing every invalid plan entry."""
        if not self.is_loaded:
            raise ModelRuntimeError("Model is not loaded")

        projections = self._target_projections(target)
        problems: list[str] = []
        for edit in plan:
            if edit.layer < 1 or edit.layer > len(self._layers):
                problems.append(
                    f"L{edit.layer} N{edit.neuron}: layer must be between 1 and "
                    f"{len(self._layers)}"
                )
                continue
            for projection in projections:
                try:
                    weight = self.projection_weight(edit.layer, projection)
                except ModelRuntimeError as error:
                    problems.append(f"L{edit.layer} N{edit.neuron}: {error}")
                    continue
                size = self._neuron_dimension(weight, projection)
                if edit.neuron >= size:
                    problems.append(
                        f"L{edit.layer} N{edit.neuron} ({projection}): neuron must "
                        f"be between 0 and {size - 1}"
                    )
        if problems:
            raise ModelRuntimeError("Invalid neuron edits:\n" + "\n".join(problems))

    def apply_plan(
        self, plan: NeuronPlan, target: str = "down_proj"
    ) -> list[AppliedEdit]:
        """Apply a plan from the original weights, replacing any previous edits."""
        if not self.is_loaded:
            raise ModelRuntimeError("Model is not loaded")
        if self._torch is None:
            raise ModelRuntimeError("Torch runtime is not initialized")

        self.validate_plan(plan, target)
        self.reset_edits()

        results: list[AppliedEdit] = []
        try:
            for edit in plan:
                for projection in self._target_projections(target):
                    weight = self.projection_weight(edit.layer, projection)
                    vector = self._vector_view(weight, projection, edit.neuron)
                    original = vector.detach().clone()
                    scale = edit.scale
                    original_norm = float(original.float().norm().item())

                    if scale == 1.0:
                        results.append(
                            AppliedEdit(
                                edit=edit,
                                projection=projection,
                                original_norm=original_norm,
                                new_norm=original_norm,
                            )
                        )
                        continue

                    new_vector = original * scale
                    with self._torch.no_grad():
                        vector.copy_(new_vector)

                    self._originals[
                        (projection, edit.layer, edit.neuron)
                    ] = original
                    results.append(
                        AppliedEdit(
                            edit=edit,
                            projection=projection,
                            original_norm=original_norm,
                            new_norm=float(new_vector.float().norm().item()),
                        )
                    )
        except Exception:
            self.reset_edits()
            raise
        return results

    def reset_edits(self) -> int:
        """Restore every edited projection vector to its original values."""
        if self._torch is None:
            return 0
        restored = 0
        for (projection, layer, neuron), original in list(self._originals.items()):
            weight = self.projection_weight(layer, projection)
            vector = self._vector_view(weight, projection, neuron)
            with self._torch.no_grad():
                vector.copy_(original)
            restored += 1
        self._originals.clear()
        return restored

    def save(self, directory: str | Path, max_shard_size: str = "5GB") -> Path:
        """Save the edited model and tokenizer."""
        if not self.is_loaded:
            raise ModelRuntimeError("Model is not loaded")
        target = Path(directory).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(
            target,
            max_shard_size=max_shard_size,
            safe_serialization=True,
        )
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(target)
        return target

    def close(self) -> None:
        """Release model references and empty the CUDA cache."""
        self.model = None
        self.tokenizer = None
        self._layers = []
        self._originals.clear()
        gc.collect()
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 - cleanup must not fail the process
                pass

    def down_proj_weight(self, layer_number: int) -> Any:
        """Return the down-projection weight matrix for a 1-based layer."""
        return self.projection_weight(layer_number, "down_proj")

    def projection_weight(self, layer_number: int, projection: str) -> Any:
        """Return an MLP projection weight matrix for a 1-based layer."""
        module = self.projection_module(layer_number, projection)
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "ndim", 0) != 2:
            raise ModelRuntimeError(
                f"Layer {layer_number} {projection} weight is not a matrix"
            )
        return weight

    def projection_module(self, layer_number: int, projection: str) -> Any:
        """Return an MLP projection module for a 1-based layer."""
        mlp = self._mlp(layer_number)

        if projection == "down_proj":
            candidates = ("down_proj", "dense_4h_to_h")
        elif projection == "up_proj":
            candidates = ("up_proj",)
        elif projection == "gate_proj":
            candidates = ("gate_proj",)
        else:
            raise ModelRuntimeError(
                f"Unknown projection {projection!r}; expected one of {PROJECTIONS}"
            )

        module = None
        for attribute in candidates:
            module = getattr(mlp, attribute, None)
            if module is not None:
                break
        if module is None:
            if projection == "down_proj" and hasattr(mlp, "c_proj"):
                raise ModelRuntimeError(
                    "GPT-2-style Conv1D MLPs are not supported by Custom Tweak yet"
                )
            raise ModelRuntimeError(
                f"Layer {layer_number} has no recognized {projection} module"
            )
        return module

    def _resolve_layers(self) -> list[Any]:
        candidates = (
            ("model", "layers"),
            ("transformer", "h"),
            ("model", "decoder", "layers"),
            ("gpt_neox", "layers"),
            ("transformer", "blocks"),
        )
        for path in candidates:
            current = self.model
            for attribute in path:
                current = getattr(current, attribute, None)
                if current is None:
                    break
            if current is not None:
                layers = list(current)
                if layers:
                    return layers
        return []

    def _layer(self, layer_number: int) -> Any:
        if layer_number < 1 or layer_number > len(self._layers):
            raise ModelRuntimeError(
                f"Layer {layer_number} is outside the valid range 1..{len(self._layers)}"
            )
        return self._layers[layer_number - 1]

    def _mlp(self, layer_number: int) -> Any:
        layer = self._layer(layer_number)
        for attribute in ("mlp", "feed_forward", "ffn"):
            mlp = getattr(layer, attribute, None)
            if mlp is not None:
                return mlp
        raise ModelRuntimeError(f"Layer {layer_number} has no recognized MLP module")

    @staticmethod
    def _target_projections(target: str) -> tuple[str, ...]:
        if target == "all":
            return PROJECTIONS
        if target not in PROJECTIONS:
            raise ModelRuntimeError(
                f"Unknown target {target!r}; expected one of {TARGETS}"
            )
        return (target,)

    @staticmethod
    def _neuron_dimension(weight: Any, projection: str) -> int:
        return int(weight.shape[1] if projection == "down_proj" else weight.shape[0])

    @staticmethod
    def _vector_view(weight: Any, projection: str, neuron: int) -> Any:
        if projection == "down_proj":
            return weight[:, neuron]
        return weight[neuron, :]

    def _dtype_candidates(self) -> list[str]:
        if self.dtype != "auto":
            return [self.dtype]
        return ["auto", "bfloat16", "float16", "float32"]

    @staticmethod
    def _dtype_kwarg_name(transformers: Any) -> str:
        """Return the dtype keyword accepted by the installed transformers."""
        version = str(getattr(transformers, "__version__", "0"))
        parts = version.split(".")
        try:
            major = int(parts[0])
            minor = int(parts[1])
        except (IndexError, ValueError):
            return "torch_dtype"
        if major >= 5 or (major == 4 and minor >= 56):
            return "dtype"
        return "torch_dtype"

    @staticmethod
    def _torch_dtype(torch: Any, name: str) -> Any:
        mapping = {
            "auto": "auto",
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if name not in mapping:
            raise ModelRuntimeError(
                f"Unsupported dtype {name!r}; use auto, float16, bfloat16, or float32"
            )
        return mapping[name]

    @staticmethod
    def _import_torch() -> Any:
        try:
            import torch
        except ModuleNotFoundError as error:
            raise ModelRuntimeError(
                "PyTorch is not installed. Install the project dependencies with "
                "'pip install -e .' and make sure your CUDA build of PyTorch is "
                "available."
            ) from error
        return torch

    @staticmethod
    def _import_transformers() -> Any:
        try:
            import transformers
        except ModuleNotFoundError as error:
            raise ModelRuntimeError(
                "transformers is not installed. Install the project dependencies "
                "with 'pip install -e .'."
            ) from error
        return transformers
