"""Model loading, GPU placement, activation capture, and generation.

Planned implementation: transformers plus accelerate loading with dtype
fallback, optional bitsandbytes 4-bit quantization, forward hooks for MLP and
attention activations, and batched greedy generation.
"""

from __future__ import annotations

# TODO(phase-1): implement ModelRuntime and activation capture.
