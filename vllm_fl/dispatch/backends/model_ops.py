# SPDX-License-Identifier: Apache-2.0
"""Lazy registrations for reusable operators required by hybrid sparse models."""

from functools import lru_cache
from importlib import import_module

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

INDEXER_OPS = (
    "validate_environment",
    "rotate_indexer_query",
    "prepare_query",
    "indexer_k_quant_and_cache",
    "gather_cache",
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
REFERENCE_INDEXER = {
    "rotate_indexer_query": "rotate_indexer_query",
    "mqa_logits": "_torch_mqa_logits",
    "topk_prefill": "topk_prefill",
    "topk_decode": "topk_decode",
    "pack_seq": "_torch_pack_seq",
    "unpack_seq": "_torch_unpack_seq",
}
MHC_OPS = {
    "mhc_pre_with_norm": "mhc_pre",
    "mhc_post": "mhc_post",
    "mhc_fused_post_pre_with_norm": "mhc_fused_post_pre",
}


def compressed_page_size(vendor, platform):
    """Storage contract; a PPU's CUDA-compatible capability is not an SM."""
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


def _register(
    registry, backend, kind, op, module, name, *, dependencies=(), guard=None
):
    from vllm_fl.utils import use_flaggems_op

    is_gems = kind == BackendImplKind.DEFAULT
    if is_gems and not all(
        use_flaggems_op(n) for n in (op, name, *(n for _, n in dependencies))
    ):
        return

    @lru_cache(None)
    def available():
        if not backend.is_available():
            return False
        try:
            for dependency_module, symbol in dependencies:
                getattr(import_module(dependency_module), symbol)
        except (ImportError, AttributeError):
            return False
        return True

    registry.register_impl(
        OpImpl(
            op_name=op,
            impl_id="default.flagos" if is_gems else "reference.torch",
            kind=kind,
            fn=lazy_op(module, name, available),
            priority=BackendPriority.DEFAULT if is_gems else BackendPriority.REFERENCE,
            supports=lazy_op(module, guard, available) if guard else None,
            allow_runtime_fallback=False,
        )
    )


def register_mhc(registry, backend, kind):
    is_gems = kind == BackendImplKind.DEFAULT
    namespace = "flaggems" if is_gems else "reference"
    module = f"vllm_fl.dispatch.backends.{namespace}.impl.mhc"
    for op, name in MHC_OPS.items():
        required = ("mhc_pre", "mhc_post") if name == "mhc_fused_post_pre" else (name,)
        dependencies = (
            tuple(("flag_gems.fused.mhc", n) for n in required) if is_gems else ()
        )
        _register(registry, backend, kind, op, module, name, dependencies=dependencies)
    activation = f"vllm_fl.dispatch.backends.{namespace}.impl." + (
        "bounded_activation" if is_gems else "mhc"
    )
    _register(
        registry,
        backend,
        kind,
        "silu_and_mul_with_clamp",
        activation,
        "silu_and_mul_with_clamp",
        dependencies=(
            (
                "flag_gems.fused.silu_and_mul_with_clamp",
                "silu_and_mul_with_clamp_kernel",
            ),
        )
        if is_gems
        else (),
        guard="supports" if is_gems else None,
    )


def register_flaggems_model_ops(registry, backend):
    specs = {
        "indexer_k_quant_and_cache": (
            "fused.indexer_k_quant_and_cache",
            "indexer_k_quant_and_cache",
            "supports_cache_write",
        ),
        "prepare_query": (
            "ops.per_token_group_quant_fp8",
            "per_token_group_quant_fp8",
            "supports_prepare_query",
        ),
        "rotate_indexer_query": (
            "ops.hadamard_transform",
            "hadamard_transform",
            "supports_rotation",
        ),
        "mqa_logits": (
            "fused.fp8_fp4_mqa_logits",
            "fp8_fp4_mqa_logits",
            "supports_mqa",
        ),
        "paged_mqa_logits": (
            "fused.bf16_paged_mqa_logits",
            "bf16_paged_mqa_logits",
            "supports_paged_mqa",
        ),
        "gather_cache": (
            "fused.cp_gather_indexer_k_quant_cache",
            "cp_gather_indexer_k_quant_cache",
            "supports_gather",
        ),
        "topk_prefill": (
            "fused.top_k_per_row_prefill",
            "top_k_per_row_prefill",
            "supports_topk",
        ),
        "topk_decode": (
            "fused.top_k_per_row_decode",
            "top_k_per_row_decode",
            "supports_topk",
        ),
        "pack_seq": ("fused.pack_seq", "pack_seq_triton", "supports_pack"),
        "unpack_seq": ("fused.unpack_seq", "unpack_seq_triton", "supports_pack"),
    }
    for name, (dependency, symbol, guard) in specs.items():
        _register(
            registry,
            backend,
            BackendImplKind.DEFAULT,
            "sparse_indexer_" + name,
            "vllm_fl.dispatch.backends.flaggems.impl.sparse_indexer",
            name,
            dependencies=(("flag_gems." + dependency, symbol),),
            guard=guard,
        )
    _register(
        registry,
        backend,
        BackendImplKind.DEFAULT,
        "fused_recurrent_kda",
        "vllm_fl.dispatch.backends.flaggems.impl.recurrent_kda",
        "fused_recurrent_kda",
        dependencies=(
            (
                "flag_gems.fused.FLA.fused_recurrent",
                "fused_recurrent_gated_delta_rule_fwd_kernel",
            ),
        ),
        guard="supports",
    )


def register_reference_indexer(registry, backend):
    module = "vllm_fl.dispatch.backends.reference.impl.sparse_indexer"
    for op, name in REFERENCE_INDEXER.items():
        _register(
            registry,
            backend,
            BackendImplKind.REFERENCE,
            "sparse_indexer_" + op,
            module,
            name,
            guard="supports_pack" if op in ("pack_seq", "unpack_seq") else None,
        )


def register_vendor_glm5(registry, backend, vendor):
    @lru_cache(None)
    def indexer():
        return import_module(
            f"vllm_fl.dispatch.backends.vendor.{vendor}.impl.sparse_indexer"
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
    # Reference decompositions are registered as REFERENCE, not hidden in a
    # vendor facade. Only real vendor bindings/fallback kernels live here.
    names = [n for n in INDEXER_OPS if vendor != "thead" or n not in REFERENCE_INDEXER]
    impls = {"sparse_indexer_" + name: make_indexer_op(name) for name in names}
    impls["sparse_indexer_capabilities"] = capabilities
    module = f"vllm_fl.dispatch.backends.vendor.{vendor}.impl.model_ops"
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
        impls[name] = lazy_op(module, name, backend.is_available)
    if vendor == "cuda":
        for op, name in MHC_OPS.items():
            impls[op] = lazy_op(module, name, backend.is_available)
    for name, fn in impls.items():
        # Upstream CUDA pack/unpack wrappers also synchronize on lengths.
        guard = (
            lazy_op(
                "vllm_fl.dispatch.backends.reference.impl.sparse_indexer",
                "supports_pack",
                backend.is_available,
            )
            if name in ("sparse_indexer_pack_seq", "sparse_indexer_unpack_seq")
            else None
        )
        registry.register_impl(
            OpImpl(
                op_name=name,
                impl_id=f"vendor.{vendor}",
                kind=BackendImplKind.VENDOR,
                fn=fn,
                vendor=vendor,
                priority=BackendPriority.VENDOR,
                supports=guard,
                allow_runtime_fallback=False,
            )
        )
