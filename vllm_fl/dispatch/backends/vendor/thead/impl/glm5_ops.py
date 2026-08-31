# SPDX-License-Identifier: Apache-2.0
"""PPU KDA and bounded activation bindings, loaded only on selection."""

from .glm5_causal_conv1d import causal_conv1d_update
from .glm5_recurrent_kda import fused_recurrent_kda
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
    "silu_and_mul_with_clamp",
]


def attention_backend():
    return "vllm_fl.dispatch.backends.flaggems.impl.glm5_attention.Glm5SparseBackend"


def silu_and_mul_with_clamp(x, limit, alpha=1.0, beta=0.0):
    if alpha != 1.0 or beta != 0.0:
        from vllm_fl.dispatch.backends.reference.impl.glm5_mhc import (
            silu_and_mul_with_clamp as ref,
        )

        return ref(x, limit, alpha, beta)
    from .glm5_activation import silu_and_mul_with_clamp as impl

    gate, up = x.chunk(2, dim=-1)
    return impl(gate, up, limit)
