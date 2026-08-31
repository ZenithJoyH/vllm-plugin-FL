# SPDX-License-Identifier: Apache-2.0
"""PPU KDA and bounded activation bindings, loaded only on selection."""

from .causal_conv1d import causal_conv1d_update
from .recurrent_kda import fused_recurrent_kda
from vllm_fl.kernels.glm5_next.safe_kda import fused_safe_kda_gate
from vllm_fl.kernels.glm5_next.portable import (
    causal_conv1d_fn,
    chunk_kda_with_safe_gate,
)

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_recurrent_kda",
    "chunk_kda_with_safe_gate",
    "fused_safe_kda_gate",
    "attention_backend",
]


def attention_backend():
    return "vllm_fl.dispatch.backends.flaggems.impl.glm5_attention.Glm5SparseBackend"
