# SPDX-License-Identifier: Apache-2.0
"""Pinned v0.24 cache reshape for GLM only; the shared runner stays unchanged."""

import torch
from dataclasses import replace
from functools import wraps
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, MambaSpec
from vllm_fl.worker.model_runner import ModelRunnerFL, _reshape_attention_kv_cache


def _reshape_kv_cache_tensors(
    self, kv_cache_raw_tensors: dict[str, torch.Tensor], kernel_block_sizes: list[int]
) -> dict[str, torch.Tensor]:
    """
    Reshape the KV cache tensors to the desired shape and dtype.

    Args:
        kv_cache_raw_tensors: The KV cache buffer of each layer, with
            correct size but uninitialized shape.
        kernel_block_sizes: The kernel block sizes for each KV cache group.
    Returns:
        Dict[str, torch.Tensor]: A map between layer names to their
        corresponding memory buffer for KV cache.
    """
    kv_caches: dict[str, torch.Tensor] = {}
    has_attn, has_mamba = (False, False)
    layer_packing: dict[str, tuple[int, int]] = {}
    for kv_tensor in self.kv_cache_config.kv_cache_tensors:
        if kv_tensor.block_stride > 0:
            for ln in kv_tensor.shared_by:
                layer_packing[ln] = (kv_tensor.offset, kv_tensor.block_stride)
    for group in self._kv_cache_spec_attn_group_iterator():
        kv_cache_spec = group.kv_cache_spec
        attn_backend = group.backend
        if group.kv_cache_group_id == len(kernel_block_sizes):
            continue
        kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
        for layer_name in group.layer_names:
            if layer_name in self.runner_only_attn_layers:
                continue
            raw_tensor = kv_cache_raw_tensors[layer_name]
            packing = layer_packing.get(layer_name)
            if packing is not None:
                _, blk_stride = packing
                num_blocks = raw_tensor.numel() // blk_stride
            else:
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
            if isinstance(kv_cache_spec, AttentionSpec):
                has_attn = True
                if kv_cache_spec.storage_block_size != kv_cache_spec.block_size:
                    storage_block_size = kv_cache_spec.storage_block_size
                    shape_block_size = 64 if storage_block_size % 64 == 0 else 32
                    assert storage_block_size % shape_block_size == 0
                    num_blocks_per_kv_block = storage_block_size // shape_block_size
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                else:
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block
                    shape_block_size = kernel_block_size
                kv_cache_shape = attn_backend.get_kv_cache_shape(
                    kernel_num_blocks,
                    shape_block_size,
                    kv_cache_spec.num_kv_heads,
                    kv_cache_spec.head_size,
                    cache_dtype_str=self.cache_config.cache_dtype,
                )
                try:
                    kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                    assert len(kv_cache_stride_order) == len(kv_cache_shape)
                except (AttributeError, NotImplementedError):
                    kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                raw_tensor = kv_cache_raw_tensors[layer_name]
                reshape_spec = kv_cache_spec
                if (
                    packing is None
                    and kv_cache_spec.page_size_padded is not None
                    and (num_blocks_per_kv_block > 1)
                ):
                    assert kv_cache_spec.page_size_bytes % num_blocks_per_kv_block == 0
                    reshape_spec = replace(
                        kv_cache_spec,
                        block_size=kernel_block_size,
                        page_size_padded=kv_cache_spec.page_size_bytes
                        // num_blocks_per_kv_block,
                    )
                kv_caches[layer_name] = _reshape_attention_kv_cache(
                    raw_tensor,
                    reshape_spec,
                    kv_cache_shape,
                    kv_cache_stride_order,
                    kernel_num_blocks,
                    packing,
                )
            elif isinstance(kv_cache_spec, MambaSpec):
                has_mamba = True
                raw_tensor = kv_cache_raw_tensors[layer_name]
                state_tensors = []
                storage_offset_bytes = 0
                for shape, dtype in zip(kv_cache_spec.shapes, kv_cache_spec.dtypes):
                    dtype_size = get_dtype_size(dtype)
                    num_element_per_page = kv_cache_spec.page_size_bytes // dtype_size
                    target_shape = (num_blocks, *shape)
                    stride = torch.empty(target_shape).stride()
                    target_stride = (num_element_per_page, *stride[1:])
                    assert storage_offset_bytes % dtype_size == 0
                    tensor = torch.as_strided(
                        raw_tensor.view(dtype),
                        size=target_shape,
                        stride=target_stride,
                        storage_offset=storage_offset_bytes // dtype_size,
                    )
                    state_tensors.append(tensor)
                    storage_offset_bytes += stride[0] * dtype_size
                kv_caches[layer_name] = state_tensors
            else:
                raise NotImplementedError
    if has_attn and has_mamba:
        self._update_hybrid_attention_mamba_layout(kv_caches, kernel_block_sizes)
    return kv_caches


def install_glm5_runner_adapter():
    original = ModelRunnerFL._reshape_kv_cache_tensors
    if getattr(original, "_glm5_only", False):
        return

    @wraps(original)
    def reshape(self, *args, **kwargs):
        if self.model_config.hf_text_config.model_type != "glm5_next_text":
            return original(self, *args, **kwargs)
        return _reshape_kv_cache_tensors(self, *args, **kwargs)

    reshape._glm5_only = True
    ModelRunnerFL._reshape_kv_cache_tensors = reshape
