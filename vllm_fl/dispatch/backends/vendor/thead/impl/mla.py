# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Thead (T-Head / PPU) FlashMLA backend.
#
# This module patches vLLM's FlashMLA support to work on PPU accelerators
# by using the flash_mla wheel (v2.0.0+ppu2.1.0, torch.ops / flash_mla_cuda)
# instead of the natively-compiled vllm._flashmla_C / vllm._flashmla_extension_C
# extensions which are not available in the empty vLLM build.
#
# vLLM 0.24.0 non-FP8 calling convention (verified):
#   scheduler_metadata, _ = get_mla_metadata(seq_lens, num_q_tokens_per_head_k,
#                                            1, is_fp8_kvcache=False)
#   o, lse = flash_mla_with_kvcache(q=..., k_cache=..., block_table=...,
#                                   cache_seqlens=..., head_dim_v=kv_lora_rank,
#                                   tile_scheduler_metadata=<FlashMLASchedMeta>,
#                                   softmax_scale=..., causal=True,
#                                   is_fp8_kvcache=False)
# The ppu2.1.0 wheel matches this exactly: get_mla_metadata returns an empty
# FlashMLASchedMeta (initialized lazily on first call) and
# flash_mla_with_kvcache ASSERTS isinstance(tile_scheduler_metadata,
# FlashMLASchedMeta) and num_splits is None.  Hence we bind the wheel
# functions DIRECTLY — the old vLLM 0.20.2 bridge that unpacked the sched
# meta into raw tensors would crash against the new wheel.
#
# FP8 MLA (get_mla_metadata_dense_fp8 / flash_mla_with_kvcache_fp8 ->
# torch.ops._flashmla_extension_C) is NOT supported on PPU; only bf16/fp16
# KV cache (is_fp8_kvcache=False) works.

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Step 1 — load the flash_mla wheel to register its custom ops
# ---------------------------------------------------------------------------
import flash_mla  # noqa: F401  — registers torch.ops.flash_mla.*
import flash_mla.flash_mla_interface as _flashmla_wheel

print("DEBUG [thead/mla.py]: flash_mla wheel imported OK", flush=True)

# ---------------------------------------------------------------------------
# Step 2 — patch the flashmla ops module on the remote
# ---------------------------------------------------------------------------
import vllm.v1.attention.ops.flashmla as _flashmla_ops_mod

# Mark flashmla as available so vLLM uses the real implementations
_flashmla_ops_mod._flashmla_C_AVAILABLE = True

# Bind the wheel implementations DIRECTLY (vLLM 0.24.0 non-FP8 convention
# matches the ppu2.1.0 wheel; no bridge wrapper needed).
_flashmla_ops_mod.flash_mla_with_kvcache = (
    _flashmla_wheel.flash_mla_with_kvcache
)
_flashmla_ops_mod.get_mla_metadata = _flashmla_wheel.get_mla_metadata
_flashmla_ops_mod.flash_mla_sparse_fwd = _flashmla_wheel.flash_mla_sparse_fwd
_flashmla_ops_mod.FlashMLASchedMeta = _flashmla_wheel.FlashMLASchedMeta

print("DEBUG [thead/mla.py]: flashmla ops patched OK", flush=True)

# ---------------------------------------------------------------------------
# Step 3 — patch is_flashmla_dense_supported / sparse_supported for PPU
# ---------------------------------------------------------------------------


def _thead_is_flashmla_dense_supported():
    return True, None


def _thead_is_flashmla_sparse_supported():
    return True, None


_flashmla_ops_mod.is_flashmla_dense_supported = _thead_is_flashmla_dense_supported
_flashmla_ops_mod.is_flashmla_sparse_supported = _thead_is_flashmla_sparse_supported

# The backend modules do "from vllm.v1.attention.ops.flashmla import ..." at
# module level — patching the ops module attribute is enough when the backend
# is imported after this module (it is: backend classes resolve lazily at
# attention selection).  Patch the backend namespaces defensively anyway.
import vllm.v1.attention.backends.mla.flashmla as _flashmla_backend_mod

_flashmla_backend_mod.is_flashmla_dense_supported = (
    _thead_is_flashmla_dense_supported
)
_flashmla_backend_mod.flash_mla_with_kvcache = (
    _flashmla_wheel.flash_mla_with_kvcache
)
_flashmla_backend_mod.get_mla_metadata = _flashmla_wheel.get_mla_metadata
_flashmla_backend_mod.FlashMLASchedMeta = _flashmla_wheel.FlashMLASchedMeta

import vllm.v1.attention.backends.mla.flashmla_sparse as _flashmla_sparse_mod

_flashmla_sparse_mod.is_flashmla_sparse_supported = (
    _thead_is_flashmla_sparse_supported
)
_flashmla_sparse_mod.flash_mla_sparse_fwd = _flashmla_wheel.flash_mla_sparse_fwd

# ---------------------------------------------------------------------------
# Step 4 — patch FlashMLABackend.supports_compute_capability for CC 8.0
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
# Step 5 — pure-PyTorch concat_and_cache_mla fallback for vendor (thead) path
# ---------------------------------------------------------------------------
# When the dispatch system selects the vendor (thead) attention backend,
# MLACommonAttentionImpl.do_kv_cache_update() calls ops.concat_and_cache_mla()
# which resolves to torch.ops._C_cache_ops.concat_and_cache_mla in vLLM's
# _custom_ops.  If the native CUDA op is not available on this platform,
# inject a pure-PyTorch alternative via _custom_ops as a fallback.
#
# vLLM 0.24.0 call signature (vllm/v1/attention/backend.py):
#   ops.concat_and_cache_mla(kv_c, k_pe.squeeze(1), kv_cache,
#                            slot_mapping.flatten(), kv_cache_dtype=...,
#                            scale=...)

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
