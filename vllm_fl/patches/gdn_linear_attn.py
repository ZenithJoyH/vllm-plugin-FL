# Copyright (c) 2026 BAAI. All rights reserved.

"""Route vLLM GDN call sites through the shared FlagOS dispatch."""

from vllm_fl.dispatch import call_op, get_default_manager, with_strict_mode


def install_gdn_dispatch() -> None:
    """Patch only GDN call sites; backend implementations own their kernels."""
    import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_mod

    if getattr(gdn_mod, "_flagos_gdn_installed", False):
        return

    native_prefill = gdn_mod.fla_chunk_gated_delta_rule
    native_decode = gdn_mod.fused_sigmoid_gating_delta_rule_update
    native_packed_decode = gdn_mod.fused_recurrent_gated_delta_rule_packed_decode

    def has_dispatch_candidate(op_name):
        manager = get_default_manager()
        manager.ensure_initialized()
        # A blacklist can remove the FlagOS implementation. On platforms
        # without a vendor candidate, preserve the captured vLLM path.
        return any(impl.is_available() for impl in manager.registry.get_implementations(op_name))

    def dispatch_prefill(
        q, k, v, g, beta, scale=None, initial_state=None,
        output_final_state=False, cu_seqlens=None, chunk_indices=None,
        chunk_offsets=None, use_qk_l2norm_in_kernel=False,
    ):
        if not has_dispatch_candidate("gdn_prefill"):
            return native_prefill(
                q, k, v, g, beta, scale=scale, initial_state=initial_state,
                output_final_state=output_final_state, cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        return call_op(
            "gdn_prefill", q=q, k=k, v=v, g=g, beta=beta,
            scale=scale, initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def dispatch_decode(
        A_log, a, b, dt_bias, q, k, v, beta=1.0, threshold=20.0,
        scale=None, initial_state=None, inplace_final_state=True,
        cu_seqlens=None, ssm_state_indices=None, num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=False, is_kda=False,
    ):
        kwargs = dict(
            A_log=A_log, a=a, b=b, dt_bias=dt_bias,
            q=q, k=k, v=v, beta=beta, threshold=threshold,
            scale=scale, initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
        )
        if is_kda or not has_dispatch_candidate("gdn_decode"):
            # KDA is a separate contract; preserve its original implementation.
            # MTP/speculative accepted-token indices are forwarded unchanged.
            return native_decode(**kwargs)
        # Decode updates the paged state in place. A failed backend may have
        # partially changed it, so never retry this call on another candidate.
        with with_strict_mode():
            return call_op("gdn_decode", **kwargs)

    def dispatch_packed_decode(
        mixed_qkv, a, b, A_log, dt_bias, scale, initial_state, out,
        ssm_state_indices, use_qk_l2norm_in_kernel=False,
    ):
        kwargs = dict(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            scale=scale, initial_state=initial_state, out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        if not has_dispatch_candidate("gdn_packed_decode"):
            return native_packed_decode(**kwargs)
        # Keep vLLM's convolution, metadata and packed fast-path selection.
        # In-place state updates must never retry after a backend exception.
        with with_strict_mode():
            return call_op("gdn_packed_decode", **kwargs)

    gdn_mod.fla_chunk_gated_delta_rule = dispatch_prefill
    gdn_mod.fused_sigmoid_gating_delta_rule_update = dispatch_decode
    gdn_mod.fused_recurrent_gated_delta_rule_packed_decode = dispatch_packed_decode
    gdn_mod._flagos_gdn_installed = True
