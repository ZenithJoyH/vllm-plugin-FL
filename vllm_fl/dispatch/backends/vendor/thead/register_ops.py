# Copyright (c) 2026 BAAI. All rights reserved.

"""
Thead (PPU) backend operator registrations.

This module registers VENDOR (thead) implementations for the dispatch system.
"""

from __future__ import annotations

import functools

from vllm_fl.dispatch.types import OpImpl, BackendImplKind, BackendPriority


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def register_builtins(registry) -> None:
    """
    Register all thead (PPU) VENDOR operator implementations.

    At registration time we also:
    - Import impl.attention to load the flash_attn_3 wheel for FA3
    - Import impl.mla to patch flashmla ops for the flash_mla wheel

    Args:
        registry: Registry to register into
    """
    # Load thead-specific patches (FA3, FlashMLA) at registration time
    # so they apply in every worker subprocess.
    from .impl import attention  # noqa: F401 — FA3 wheel + reshape_and_cache
    from .impl import mla  # noqa: F401 — flash_mla wheel patches

    from .thead import TheadBackend

    backend = TheadBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="attention_backend",
            impl_id="vendor.thead",
            kind=BackendImplKind.VENDOR,
            fn=_bind_is_available(backend.attention_backend, is_avail),
            vendor="thead",
            priority=BackendPriority.VENDOR,
        ),
    ]

    registry.register_many(impls)
