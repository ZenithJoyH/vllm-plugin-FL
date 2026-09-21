# SPDX-License-Identifier: Apache-2.0
"""Native MetaX cache writes; fallback selection belongs to the dispatcher."""


def native_cache_available() -> bool:
    import torch

    from vllm import _custom_ops  # noqa: F401 -- loads native operator registrations

    try:
        # A Python wrapper or schema alone is not sufficient in an empty wheel.
        return torch._C._dispatch_has_kernel_for_dispatch_key(
            "_C_cache_ops::reshape_and_cache_flash", "CUDA"
        )
    except RuntimeError:
        return False


def reshape_and_cache_flash_maca(
    key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
):
    from vllm import _custom_ops

    return _custom_ops.reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
