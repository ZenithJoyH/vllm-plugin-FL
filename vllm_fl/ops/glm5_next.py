# SPDX-License-Identifier: Apache-2.0
"""GLM layer contracts, independent of vendor kernels and CustomOp patches."""

from torch import nn
from vllm_fl.dispatch import CachedOp

causal_conv1d_fn = CachedOp("causal_conv1d_fn")
causal_conv1d_update = CachedOp("causal_conv1d_update")
fused_recurrent_kda = CachedOp("fused_recurrent_kda")
chunk_kda_with_safe_gate = CachedOp("chunk_kda_with_safe_gate")
fused_safe_kda_gate = CachedOp("fused_safe_kda_gate")


class MHCPreOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_pre_with_norm"))


class MHCPostOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_post"))


class MHCFusedPostPreOp(nn.Module):
    forward = staticmethod(CachedOp("mhc_fused_post_pre_with_norm"))


class SiluAndMulWithClamp(nn.Module):
    def __init__(self, swiglu_limit, alpha=1.0, beta=0.0, **kwargs):
        super().__init__()
        self.limit, self.alpha, self.beta = swiglu_limit, alpha, beta
        self.op = CachedOp("silu_and_mul_with_clamp")

    def forward(self, x):
        return self.op(x, self.limit, self.alpha, self.beta)
