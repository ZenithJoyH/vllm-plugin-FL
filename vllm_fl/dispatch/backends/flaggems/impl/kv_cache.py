# SPDX-License-Identifier: Apache-2.0
"""FlagGems implementations of paged-cache operations."""


def reshape_and_cache_flash_flaggems(
    key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale
):
    from flag_gems import reshape_and_cache_flash

    return reshape_and_cache_flash(
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
