# SPDX-License-Identifier: Apache-2.0
"""FlagGems MHC adapters; the contract includes exactly one RMSNorm."""

from vllm_fl.dispatch.backends.reference.impl.glm5_mhc import rms_norm


def mhc_pre(
    residual,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    norm_weight=None,
    norm_eps=0.0,
):
    from flag_gems.fused.mhc import mhc_pre as impl

    post, comb, x = impl(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        n_splits,
    )
    return post, comb, rms_norm(x, norm_weight, norm_eps)


def mhc_post(*args, **kwargs):
    from flag_gems.fused.mhc import mhc_post as impl

    return impl(*args, **kwargs)


def mhc_fused_post_pre(
    x,
    residual,
    post_layer_mix,
    comb_res_mix,
    fn,
    hc_scale,
    hc_base,
    rms_eps,
    hc_pre_eps,
    hc_sinkhorn_eps,
    hc_post_mult_value,
    sinkhorn_repeat,
    n_splits=1,
    tile_n=1,
    norm_weight=None,
    norm_eps=0.0,
):
    updated = mhc_post(x, residual, post_layer_mix, comb_res_mix)
    return (
        updated,
        *mhc_pre(
            updated,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        ),
    )


def silu_and_mul_with_clamp(x, limit, alpha=1.0, beta=0.0):
    if alpha != 1.0 or beta != 0.0:
        from vllm_fl.dispatch.backends.reference.impl.glm5_mhc import (
            silu_and_mul_with_clamp as ref,
        )

        return ref(x, limit, alpha, beta)
    from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp as impl

    gate, up = x.chunk(2, dim=-1)
    return impl(gate, up, limit)
