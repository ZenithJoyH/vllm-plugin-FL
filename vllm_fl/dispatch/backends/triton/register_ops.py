# Copyright (c) 2026 BAAI. All rights reserved.
"""Register portable plugin kernels without importing model or device modules."""

from importlib import import_module

from ...types import BackendImplKind, BackendPriority, OpImpl


def _triton_available():
    from vllm.triton_utils import HAS_TRITON

    return HAS_TRITON


def _lazy_op(module, name):
    def invoke(*args, **kwargs):
        return getattr(import_module(module), name)(*args, **kwargs)

    invoke.__name__ = name
    invoke._is_available = _triton_available
    return invoke


def register_builtins(registry):
    # Operator names encode model-specific semantics, not the hardware vendor.
    # These ABIs (paged QSA, PLE row I/O, packed FP32-beta decode) are missing
    # from FlagGems 4df52d9. Vendor backends may override them via normal policy.
    groups = {
        "qsa": [
            "qsa_mqa_paged",
            "expand_qsa_block_indices",
            "qsa_select_paged_tokens",
            "qsa_sparse_paged_attention",
            "qsa_store_cache_rows",
            "qsa_compress_groups_with_ratio",
        ],
        "ple_state": ["ple_state_gather", "ple_state_scatter_"],
        "gdn": ["gdn_packed_decode"],
        "slot_mapping": ["compute_common_slot_mapping"],
    }
    for module, names in groups.items():
        for name in names:
            registry.register_impl(
                OpImpl(
                    op_name="qwen38_" + name,
                    impl_id="default.triton.qwen38",
                    kind=BackendImplKind.DEFAULT,
                    priority=BackendPriority.DEFAULT,
                    fn=_lazy_op(
                        "vllm_fl.dispatch.backends.triton.qwen38." + module, name
                    ),
                )
            )
