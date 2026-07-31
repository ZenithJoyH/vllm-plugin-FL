# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Thead (T-Head / PPU) FlashMLA backend.
#
# This module patches vLLM's FlashMLA support to work on PPU accelerators
# by using the flash_mla wheel (torch.ops.flash_mla.*) instead of the
# natively-compiled vllm._flashmla_C extension which is not available.
#
# At module load time we:
#   1. Import flash_mla to register its custom ops.
#   2. Alias torch.ops._flashmla_C -> torch.ops.flash_mla so that vLLM's
#      existing flashmla code paths can resolve the ops.
#   3. Patch the vllm.v1.attention.ops.flashmla module to mark flashmla as
#      available and replace stub functions with real ones from the wheel.
#   4. Patch is_flashmla_dense_supported / is_flashmla_sparse_supported to
#      return True on PPU.
#   5. Patch FlashMLABackend.supports_compute_capability to allow CC 8.0
#   6. Inject pure-PyTorch concat_and_cache_mla fallback into _custom_ops
#      for the vendor (thead) attention backend path.

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Step 1 — load the flash_mla wheel to register its custom ops
# ---------------------------------------------------------------------------
import flash_mla  # noqa: F401  — registers torch.ops.flash_mla.*
import flash_mla.flash_mla_interface  # noqa: F401  — real functions

print("DEBUG [thead/mla.py]: flash_mla wheel imported OK", flush=True)

# ---------------------------------------------------------------------------
# Step 2 — alias torch.ops._flashmla_C -> torch.ops.flash_mla
# ---------------------------------------------------------------------------
torch.ops._flashmla_C = torch.ops.flash_mla

# ---------------------------------------------------------------------------
# Step 3 — patch the flashmla ops module on the remote
# ---------------------------------------------------------------------------
import vllm.v1.attention.ops.flashmla as _flashmla_ops_mod

# Mark flashmla as available so vLLM uses real implementations
_flashmla_ops_mod._flashmla_C_AVAILABLE = True

# Replace stub functions with real ones from the wheel
# BUT: wheel and vLLM have different calling conventions for
# flash_mla_with_kvcache — vLLM may pass tile_scheduler_metadata as a
# FlashMLASchedMeta object (non-FP8 path) that bundles both
# tile_scheduler_metadata and num_splits, while the wheel expects them
# as two separate arguments.  We provide a bridge wrapper.
_flashmla_wheel_impl = flash_mla.flash_mla_interface.flash_mla_with_kvcache


def _thead_flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: object,
    num_splits: object = None,   # optional, may be baked into metadata
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bridge between vLLM and wheel calling conventions.

    vLLM may call flash_mla_with_kvcache in two ways:
    1. FP8 path: passes tile_scheduler_metadata and num_splits as separate args
    2. Non-FP8 path: passes a FlashMLASchedMeta object as tile_scheduler_metadata
       (no explicit num_splits) — need to extract both fields.
    """
    if num_splits is None:
        # Non-FP8 path: tile_scheduler_metadata is a FlashMLASchedMeta
        _meta = tile_scheduler_metadata
        if hasattr(_meta, "tile_scheduler_metadata") and hasattr(_meta, "num_splits"):
            tile_scheduler_metadata = _meta.tile_scheduler_metadata
            num_splits = _meta.num_splits
        else:
            num_splits = torch.tensor(0, device=q.device, dtype=torch.int32)

    # Wheel kernel enforces kMaxSplits (typically 64).  vLLM's metadata
    # builder may produce larger values depending on cache lengths.
    if isinstance(num_splits, torch.Tensor) and num_splits.numel() > 0:
        num_splits = num_splits.clamp(max=63)

    return _flashmla_wheel_impl(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
    )


_flashmla_ops_mod.flash_mla_with_kvcache = _thead_flash_mla_with_kvcache
_flashmla_ops_mod.get_mla_metadata = flash_mla.flash_mla_interface.get_mla_metadata
_flashmla_ops_mod.flash_mla_sparse_fwd = flash_mla.flash_mla_interface.flash_mla_sparse_fwd
_flashmla_ops_mod.FlashMLASchedMeta = flash_mla.flash_mla_interface.FlashMLASchedMeta

print("DEBUG [thead/mla.py]: flashmla ops patched OK", flush=True)

# ---------------------------------------------------------------------------
# Step 4 — patch is_flashmla_dense_supported / sparse_supported for PPU
# ---------------------------------------------------------------------------

def _thead_is_flashmla_dense_supported():
    return True, None


def _thead_is_flashmla_sparse_supported():
    return True, None


_flashmla_ops_mod.is_flashmla_dense_supported = _thead_is_flashmla_dense_supported
_flashmla_ops_mod.is_flashmla_sparse_supported = _thead_is_flashmla_sparse_supported

# Important: also patch the backend module's namespace because flashmla.py
# does "from vllm.v1.attention.ops.flashmla import is_flashmla_dense_supported"
# at module level — reassigning the ops module's attribute is not enough.
import vllm.v1.attention.backends.mla.flashmla as _flashmla_backend_mod
_flashmla_backend_mod.is_flashmla_dense_supported = _thead_is_flashmla_dense_supported

import vllm.v1.attention.backends.mla.flashmla_sparse as _flashmla_sparse_mod
_flashmla_sparse_mod.is_flashmla_sparse_supported = _thead_is_flashmla_sparse_supported

# ---------------------------------------------------------------------------
# Step 5 — patch FlashMLABackend.supports_compute_capability for CC 8.0
# ---------------------------------------------------------------------------
from vllm.v1.attention.backends.mla.flashmla import FlashMLABackend
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
from vllm.platforms.interface import DeviceCapability


def _thead_flashmla_supports_cc(cls, capability: DeviceCapability) -> bool:
    # PPU CC = 8.0 — need to accept it in addition to Hopper (9.x)
    return capability.major in [8, 9, 10]


FlashMLABackend.supports_compute_capability = classmethod(_thead_flashmla_supports_cc)
FlashMLASparseBackend.supports_compute_capability = classmethod(_thead_flashmla_supports_cc)

print("DEBUG [thead/mla.py]: FlashMLABackend patched for CC 8.0 OK", flush=True)

# ---------------------------------------------------------------------------
# Step 6 — pure-PyTorch concat_and_cache_mla fallback for vendor (thead) path
# ---------------------------------------------------------------------------
# When the dispatch system selects the vendor (thead) attention backend,
# FlashMLABackendImpl.do_kv_cache_update() calls ops.concat_and_cache_mla()
# which resolves to torch.ops._C_cache_ops.concat_and_cache_mla.
# If the native CUDA op is not available on this platform, inject a
# pure-PyTorch alternative via _custom_ops as a fallback.

_concat_and_cache_mla_cuda_available = False
try:
    _test_kv_c = torch.empty(1, 64, device="cuda")
    _test_k_pe = torch.empty(1, 128, device="cuda")
    _test_cache = torch.empty(8, 128, 192, device="cuda")
    _test_sm = torch.zeros(1, dtype=torch.int64, device="cuda")
    _test_scale = torch.ones(1, device="cuda")
    torch.ops._C_cache_ops.concat_and_cache_mla(
        _test_kv_c, _test_k_pe, _test_cache, _test_sm, "auto", _test_scale,
    )
    _concat_and_cache_mla_cuda_available = True
    del _test_kv_c, _test_k_pe, _test_cache, _test_sm, _test_scale
    print("DEBUG [thead/mla.py]: using CUDA _C_cache_ops.concat_and_cache_mla",
          flush=True)
except Exception:
    print("DEBUG [thead/mla.py]: CUDA concat_and_cache_mla unavailable, "
          "using pure-PyTorch fallback", flush=True)


def _thead_concat_and_cache_mla(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    scale: torch.Tensor,
) -> None:
    """KV cache write for MLA, with CUDA native + pure-PyTorch fallback.

    Used by the vendor (thead) attention backend path.  Tries the native
    CUDA custom op first; falls back to pure-PyTorch scatter_ if the .so
    is not available on this platform.
    """
    if _concat_and_cache_mla_cuda_available:
        torch.ops._C_cache_ops.concat_and_cache_mla(
            kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale,
        )
        return

    # ---- pure-PyTorch fallback ----
    del kv_cache_dtype, scale  # unused in pure-torch path

    if kv_cache.numel() == 0:
        return

    kv_channels = kv_c.shape[-1]
    total_dim = kv_cache.shape[-1]

    # Flatten cache to 2D: [num_blocks * block_size, total_dim]
    cache_flat = kv_cache.reshape(-1, total_dim)

    # slot_mapping defines how many actual tokens we process.
    # kv_c may be block-aligned (larger first dim) — slice to actual tokens.
    num_tokens = slot_mapping.shape[0]
    kv_c = kv_c[:num_tokens]
    k_pe = k_pe[:num_tokens]

    # Map -1 to 0 and zero out padding tokens' data
    valid_mask = (slot_mapping >= 0).to(kv_c.dtype).view(-1, 1)
    safe_slots = slot_mapping.clamp(min=0)

    # Write kv_c into cache_flat[:, :kv_channels] via scatter_
    k_flat = cache_flat[:, :kv_channels]
    masked_kv_c = kv_c * valid_mask
    dst = safe_slots.view(-1, 1).expand(-1, kv_channels)
    k_flat.scatter_(0, dst, masked_kv_c)

    # Write k_pe into cache_flat[:, kv_channels:] via scatter_
    if k_pe.numel() > 0:
        pe_dim = k_pe.shape[-1]
        pe_flat = cache_flat[:, kv_channels:]
        masked_k_pe = k_pe * valid_mask
        dst_pe = safe_slots.view(-1, 1).expand(-1, pe_dim)
        pe_flat.scatter_(0, dst_pe, masked_k_pe)


# Inject into _custom_ops so that the vendor attention backend's
# do_kv_cache_update can find it via ops.concat_and_cache_mla().
import vllm._custom_ops as _custom_ops_mod
_custom_ops_mod.concat_and_cache_mla = _thead_concat_and_cache_mla

print(
    "DEBUG [thead/mla.py]: concat_and_cache_mla patched with "
    "PyTorch fallback for vendor path OK",
    flush=True,
)
