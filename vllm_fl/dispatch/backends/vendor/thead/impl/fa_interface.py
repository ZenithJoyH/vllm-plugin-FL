# Copyright (c) 2023, Tri Dao.  Ported from the PPU (T-Head) vLLM build
# (vllm.vllm_flash_attn.flash_attn_interface), which is itself derived from
# https://github.com/vllm-project/flash-attention.
#
# Thead (T-Head / PPU) flash attention interface.
#
# This module is a port of ppu-native's ``vllm/vllm_flash_attn/flash_attn_interface.py``
# (vLLM 0.23.0 + ppu2.1.0).  It wraps the PPU FA2 / FA3 wheels:
#
#   - FA2: ``torch.ops.flash_attn._flash_attn_varlen_forward``
#   - FA3: ``torch.ops.flash_attn_3.fwd`` (35-arg signature)
#   - FA3 scheduler metadata: ``torch.ops.flash_attn_3.get_scheduler_metadata``
#
# It is used by impl/attention.py to patch the vLLM 0.24.0
# ``vllm.v1.attention.backends.{fa_utils,flash_attn}`` module namespaces so
# that ``FlashAttentionImpl`` / ``FlashAttentionMetadataBuilder`` can resolve
# these names on the PPU platform (where ``current_platform.is_cuda()`` is
# False and vLLM's own compiled FA extensions are absent).
#
# PPU notes preserved from the original port:
#   - FA2 does not support writing in-place -> ``out.copy_(out_fa2)``.
#   - FA3 uses ``max_seqlen_k`` to choose the tile size; when
#     ``cu_seqlens_k is None`` (paged decode) ``max_seqlen_k`` must be 1
#     (Aone#75639039).

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# Wheel availability probes (do NOT import vllm.vllm_flash_attn — it raises
# ImportError in the empty vLLM build).
# ---------------------------------------------------------------------------
try:
    import flash_attn  # noqa: F401  — registers torch.ops.flash_attn.*
    FA2_UNAVAILABLE_REASON = None
    FA2_AVAILABLE = True
except ImportError as e:
    FA2_UNAVAILABLE_REASON = str(e)
    FA2_AVAILABLE = False

try:
    import flash_attn_3._C  # noqa: F401  — registers torch.ops.flash_attn_3.*
    FA3_UNAVAILABLE_REASON = None
    FA3_AVAILABLE = True
except ImportError as e:
    FA3_UNAVAILABLE_REASON = str(e)
    FA3_AVAILABLE = False

DEFAULT_FA_VERSION = 2


def _is_fa2_supported(device=None) -> Tuple[bool, Optional[str]]:
    if not FA2_AVAILABLE:
        return False, f"FA2 is unavailable due to: {FA2_UNAVAILABLE_REASON}"
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False, (
            "FA2 is only supported on devices with compute capability >= 8"
        )
    return True, None


def _is_fa3_supported(device=None) -> Tuple[bool, Optional[str]]:
    if not FA3_AVAILABLE:
        return False, f"FA3 is unavailable due to: {FA3_UNAVAILABLE_REASON}"
    cap = torch.cuda.get_device_capability(device)
    if (
        cap[0] < 8
        or cap[0] >= 10
        or cap == (8, 6)
    ):
        return False, (
            "FA3 is only supported on devices with compute capability >= 8 "
            "excluding 8.6 and Blackwell archs (>=10)"
        )
    return True, None


def is_fa_version_supported(fa_version: int, device=None) -> bool:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[0]
    elif fa_version == 3:
        return _is_fa3_supported(device)[0]


def fa_version_unsupported_reason(fa_version: int, device=None) -> Optional[str]:
    assert fa_version in [2, 3], f"Unsupported FA version: {fa_version}"
    if fa_version == 2:
        return _is_fa2_supported(device)[1]
    elif fa_version == 3:
        return _is_fa3_supported(device)[1]


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
) -> Optional[int]:
    """Return the FA version to use on PPU (matches vLLM 0.24.0 kwargs)."""
    del head_size, head_size_v, has_sinks  # PPU FA3 has no such restrictions
    if requires_alibi:
        # FA3 does not support ALiBi
        return 2 if FA2_AVAILABLE else None
    if FA3_AVAILABLE and is_fa_version_supported(3):
        return 3
    if FA2_AVAILABLE and is_fa_version_supported(2):
        return 2
    return None


#
#  For vLLM we only care about `flash_attn_varlen_func` and
#  `get_scheduler_metadata`.
#


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# NOTE only used in FA3
def get_scheduler_metadata(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv, headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    has_softcap=False,
    num_splits=0,    # Can be tuned for speed
    pack_gqa=None,   # Can be tuned for speed
    sm_margin=0,     # Can be tuned if some SMs are used for communication
):
    cache_seqlens = maybe_contiguous(cache_seqlens)
    if headdim_v is None:
        headdim_v = headdim
    scheduler_metadata = torch.ops.flash_attn_3.get_scheduler_metadata(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv,
        headdim, headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0], window_size[1],
        0,  # attention_chunk
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    return scheduler_metadata


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,  # only used for non-paged prefill
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: Optional[List[int]] = None,
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    # FA3 Only
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    # Version selector
    fa_version: int = DEFAULT_FA_VERSION,
    s_aux=None,
    # vLLM 0.24.0 extra kwargs (FA4-only features; not supported by PPU FA3)
    output_scale=None,
    mask_mod=None,
    aux_tensors=None,
    dynamic_causal=None,
):
    """Custom flash_attn_varlen_func for PPU using the FA2/FA3 wheels.

    Accepts the vLLM 0.24.0 calling convention (keyword args), including the
    FA4-only kwargs ``dynamic_causal`` / ``mask_mod`` / ``aux_tensors`` /
    ``output_scale`` which are rejected if non-None.
    """
    if output_scale is not None or mask_mod is not None or aux_tensors is not None:
        raise NotImplementedError(
            "Thead FA3 does not support output_scale / mask_mod / aux_tensors"
        )
    if dynamic_causal is not None:
        raise NotImplementedError(
            "Thead FA3 does not support dynamic_causal (per-sequence causal)"
        )
    assert cu_seqlens_k is not None or seqused_k is not None, \
        "cu_seqlens_k or seqused_k must be provided"
    assert cu_seqlens_k is None or seqused_k is None, \
        "cu_seqlens_k and seqused_k cannot be provided at the same time"
    assert block_table is None or seqused_k is not None, \
        "seqused_k must be provided if block_table is provided"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    # custom op does not support non-tuple input
    real_window_size: Tuple[int, int]
    if window_size is None:
        real_window_size = (-1, -1)
    else:
        assert len(window_size) == 2
        real_window_size = (window_size[0], window_size[1])
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    dummy_cu_seqlens_k = torch.empty_like(cu_seqlens_q)

    if fa_version == 2:
        if (
            scheduler_metadata is not None
            and q_descale is not None
            and k_descale is not None
            and v_descale is not None
        ):
            raise NotImplementedError(
                "FA2 does not support scheduler_metadata, q_descale, "
                "k_descale, v_descale"
            )
        if num_splits > 1:
            raise NotImplementedError("FA2 does not support num_splits > 1")
        if s_aux is not None:
            raise NotImplementedError("FA2 does not support s_aux")
        out_fa2, softmax_lse, _, _ = torch.ops.flash_attn._flash_attn_varlen_forward(
            q, k, v,
            cu_seqlens_q,
            # cu_seqlens_k not used since we use seqused_k, but flash_api.cpp
            # still wants it so we pass all zeros
            dummy_cu_seqlens_k if cu_seqlens_k is None else cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=real_window_size[0],
            window_size_right=real_window_size[1],
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            return_softmax=(return_softmax_lse and dropout_p > 0),
            block_table=block_table,
            leftpad_k=None,
            seqused_k=seqused_k,
            zero_tensors=False,
        )
        # PPU NOTE: ppu fa2 api does not support inplace,
        # so it needs to be done in framework.
        if out is not None:
            out.copy_(out_fa2)
        else:
            out = out_fa2
    elif fa_version == 3:
        assert alibi_slopes is None, "Alibi is not supported in FA3"
        # [Note]: Aone#75639039
        # PPU FA3 use max_seqlen_k to choose tile
        # max_seqlen_k is needed when cu_seqlens_k on prefill
        if cu_seqlens_k is None:
            max_seqlen_k = 1

        out, softmax_lse, _, _ = torch.ops.flash_attn_3.fwd(
            q, k, v,
            None, None,       # k_new, v_new
            q_v,
            out,
            cu_seqlens_q,
            cu_seqlens_k,     # cu_seqlens_k
            None,             # cu_seqlens_k_new
            None, seqused_k,  # seqused_q, seqused_k
            max_seqlen_q, max_seqlen_k,
            block_table,
            None,             # kv_batch_idx
            None,             # leftpad_k
            None, None, None,  # rotary_cos, rotary_sin, seqlens_rotary
            q_descale, k_descale, v_descale,
            softmax_scale,
            causal,
            real_window_size[0], real_window_size[1],
            0,                # attention chunk
            softcap,
            True,             # rotary_interleaved
            scheduler_metadata,
            num_splits,
            None,             # pack_gqa
            0,                # sm_margin
            s_aux,            # s_aux
        )
    else:
        raise ValueError(f"Unsupported FA version: {fa_version}")
    return (out, softmax_lse) if return_softmax_lse else out
