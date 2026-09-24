# SPDX-License-Identifier: Apache-2.0
"""Select paged-cache writes independently of the attention backend."""

from vllm_fl.dispatch import CachedOp

reshape_and_cache_flash = CachedOp("reshape_and_cache_flash")
