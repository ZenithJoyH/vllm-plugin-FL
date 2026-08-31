# SPDX-License-Identifier: Apache-2.0
"""Bind FlagGems' vector-gated recurrent kernel, not its scalar GDN wrapper.

Pinned FlagGems 5.3.5 fused/FLA/fused_recurrent.py has IS_KDA in the
generic kernel, while its public GDN wrapper fixes IS_KDA=False. Keep all
recurrence math in FlagGems; only launch/contract adaptation lives here.
Active scheduler state IDs must be valid and distinct; empty sequences may
carry sentinel IDs. Speculative and non-contiguous-state layouts are excluded.
"""

import torch
import triton


def supports(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    **kwargs,
):
    if kwargs or any(
        t is None for t in (beta, initial_state, cu_seqlens, ssm_state_indices)
    ):
        return False
    return (
        q.device.type == "cuda"
        and q.ndim == 4
        and q.shape[0] == 1
        and q.shape == k.shape == g.shape
        and v.shape[:3] == q.shape[:3]
        and q.shape[2:] == (4, 128)
        and v.shape[-1] == 128
        and beta.shape == q.shape[:3]
        and beta.is_contiguous()
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and g.dtype == initial_state.dtype == torch.float32
        and initial_state.shape[1:] == (4, 128, 128)
        and initial_state.is_contiguous()
        and initial_state.shape[0] > 0
        and beta.dtype in (torch.float32, torch.bfloat16)
        and inplace_final_state
        and cu_seqlens.ndim == 1
        and cu_seqlens.is_contiguous()
        and cu_seqlens.dtype in (torch.int32, torch.int64)
        and cu_seqlens.numel() == q.shape[1] + 1
        and ssm_state_indices.ndim in (1, 2)
        and ssm_state_indices.dtype in (torch.int32, torch.int64)
        and (ssm_state_indices.ndim == 1 or ssm_state_indices.shape[1] >= 1)
        and ssm_state_indices.shape[0] >= q.shape[1]
        and all(
            t.device == q.device
            for t in (k, v, g, beta, initial_state, cu_seqlens, ssm_state_indices)
        )
    )


def fused_recurrent_kda(
    q,
    k,
    v,
    g,
    beta=None,
    scale=None,
    initial_state=None,
    inplace_final_state=True,
    use_qk_l2norm_in_kernel=True,
    cu_seqlens=None,
    ssm_state_indices=None,
    **kwargs,
):
    from flag_gems.fused.FLA.fused_recurrent import (
        fused_recurrent_gated_delta_rule_fwd_kernel,
    )

    if not supports(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        inplace_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        ssm_state_indices,
        **kwargs,
    ):
        raise ValueError("Unsupported FlagGems recurrent KDA contract")
    # The existing kernel indexes dense q/k/v/g. Copies preserve dtype and
    # are graph-pool allocations; no host tensor-value reads are performed.
    q, k, v, g = (t.contiguous() for t in (q, k, v, g))
    output = torch.zeros_like(v)  # empty/padded sequences return deterministic 0
    tokens, heads, width = q.shape[1:]
    fused_recurrent_gated_delta_rule_fwd_kernel[(1, 4, tokens * heads)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=output,
        h0=initial_state,
        ht=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
        scale=width**-0.5 if scale is None else scale,
        N=tokens,
        T=tokens,
        B=1,
        H=heads,
        HV=heads,
        K=width,
        V=width,
        BK=triton.next_power_of_2(width),
        BV=32,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_indices_tok=1,
        IS_BETA_HEADWISE=False,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=True,
        IS_KDA=True,
        num_warps=4,
        num_stages=1,
    )
    return output, initial_state
