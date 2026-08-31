# SPDX-License-Identifier: Apache-2.0
"""NVIDIA bindings; no PPU imports or process-start provider switches."""

from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.fla.ops.kda import fused_recurrent_kda
from vllm_fl.kernels.glm5_next.safe_kda import (
    chunk_kda_with_safe_gate,
    fused_safe_kda_gate,
)

__all__ = [
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "fused_recurrent_kda",
    "chunk_kda_with_safe_gate",
    "fused_safe_kda_gate",
    "attention_backend",
    "mhc_pre",
    "mhc_post",
    "mhc_fused_post_pre",
]


def attention_backend():
    return "vllm_fl.dispatch.backends.vendor.cuda.impl.glm5_attention.Glm5SparseBackend"


def mhc_pre(*args, **kwargs):
    from vllm.model_executor.layers.mhc import MHCPreOp

    return MHCPreOp.forward_cuda(None, *args, **kwargs)


def mhc_post(*args, **kwargs):
    from vllm.model_executor.layers.mhc import MHCPostOp

    return MHCPostOp.forward_cuda(None, *args, **kwargs)


def mhc_fused_post_pre(*args, **kwargs):
    from vllm.model_executor.layers.mhc import MHCFusedPostPreOp

    return MHCFusedPostPreOp.forward_cuda(None, *args, **kwargs)
