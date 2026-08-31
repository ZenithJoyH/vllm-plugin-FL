# SPDX-License-Identifier: Apache-2.0
"""Lazy GLM registrations used by the existing backend discovery lifecycle."""

from functools import lru_cache
from importlib import import_module
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

INDEXER_OPS = (
    "validate_environment",
    "rotate_indexer_query",
    "per_token_group_quant_fp8",
    "indexer_k_quant_and_cache",
    "cp_gather_indexer_k_quant_cache",
    "mqa_logits",
    "paged_mqa_logits",
    "topk_prefill",
    "topk_decode",
    "pack_seq",
    "unpack_seq",
    "persist_prefill_tail",
    "kpool_compress_and_write_cache",
    "kpool_decode_update_and_maybe_write_cache_batched",
    "expand_pools_to_tokens",
    "append_tail_to_topk",
)


def compressed_page_size(vendor, platform):
    """Storage page constraint; PPU's CUDA-compatible capability is not an SM."""
    if vendor == "thead":
        return 32
    if vendor == "cuda":
        capability = platform.get_device_capability()
        if capability is None:
            raise RuntimeError("GLM CUDA indexer requires a device capability")
        return 64 if capability.major < 10 else 32
    raise NotImplementedError(f"No GLM indexer page contract for {vendor}")


def lazy_op(module, name, available):
    @lru_cache(None)
    def resolve():
        return getattr(import_module(module), name)

    def invoke(*args, **kwargs):
        return resolve()(*args, **kwargs)

    invoke._is_available = available
    return invoke


def register_mhc(registry, backend, kind):
    from vllm_fl.utils import use_flaggems_op

    is_gems = kind == BackendImplKind.DEFAULT
    namespace = "flaggems" if is_gems else "reference"
    module = f"vllm_fl.dispatch.backends.{namespace}.impl.glm5_mhc"
    names = {
        f"glm5_{name}": name for name in ("mhc_pre", "mhc_post", "mhc_fused_post_pre")
    }
    names["silu_and_mul_with_clamp"] = "silu_and_mul_with_clamp"
    for op, name in names.items():
        if is_gems and name == "silu_and_mul_with_clamp":
            from vllm.platforms import current_platform

            if getattr(current_platform, "vendor_name", None) == "thead":
                # The pinned public wrapper creates a scalar during capture;
                # PPU uses its prewarmed, graph-safe wrapper of the same kernel.
                continue
        required = ("mhc_pre", "mhc_post") if name == "mhc_fused_post_pre" else (name,)
        if is_gems and (
            not use_flaggems_op(op) or not all(use_flaggems_op(n) for n in required)
        ):
            continue
        registry.register_impl(
            OpImpl(
                op_name=op,
                impl_id="default.flagos" if is_gems else "reference.torch",
                kind=kind,
                fn=lazy_op(module, name, backend.is_available),
                vendor=None,
                priority=BackendPriority.DEFAULT
                if is_gems
                else BackendPriority.REFERENCE,
            )
        )


def register_vendor_glm5(registry, backend, vendor):
    @lru_cache(None)
    def indexer():
        return import_module(
            f"vllm_fl.dispatch.backends.vendor.{vendor}.impl.glm5_indexer"
        ).IndexerOps()

    def make_indexer_op(name):
        def invoke(*args, **kwargs):
            return getattr(indexer(), name)(*args, **kwargs)

        invoke._is_available = backend.is_available
        return invoke

    def capabilities():
        import torch
        from vllm.platforms import current_platform

        return {
            "cache_dtype": torch.uint8 if vendor == "cuda" else torch.bfloat16,
            "query_dtype": current_platform.fp8_dtype()
            if vendor == "cuda"
            else torch.bfloat16,
            "compressed_page_size": compressed_page_size(vendor, current_platform),
        }

    capabilities._is_available = backend.is_available
    impls = {f"glm5_indexer_{name}": make_indexer_op(name) for name in INDEXER_OPS}
    impls["glm5_indexer_capabilities"] = capabilities
    module = f"vllm_fl.dispatch.backends.vendor.{vendor}.impl.glm5_ops"
    impls["glm5_attention_backend"] = lazy_op(
        module, "attention_backend", backend.is_available
    )
    for name in (
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "fused_recurrent_kda",
        "chunk_kda_with_safe_gate",
        "fused_safe_kda_gate",
    ):
        impls["glm5_" + name] = lazy_op(module, name, backend.is_available)
    if vendor == "cuda":
        for name in ("mhc_pre", "mhc_post", "mhc_fused_post_pre"):
            impls["glm5_" + name] = lazy_op(module, name, backend.is_available)
    if vendor == "thead":
        impls["silu_and_mul_with_clamp"] = lazy_op(
            module, "silu_and_mul_with_clamp", backend.is_available
        )
    registry.register_many(
        [
            OpImpl(
                op_name=name,
                impl_id=f"vendor.{vendor}",
                kind=BackendImplKind.VENDOR,
                fn=fn,
                vendor=vendor,
                priority=BackendPriority.VENDOR,
            )
            for name, fn in impls.items()
        ]
    )
