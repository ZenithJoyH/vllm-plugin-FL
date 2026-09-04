# Copyright (c) 2026 BAAI. All rights reserved.

"""T-Head-only initialization before vLLM fallback schemas are registered."""

import os


def initialize_native_extensions() -> None:
    """Load the PPU bundle without consulting the unfinished current_platform.

    The native loader owns SO paths, load order, idempotency and vLLM module
    compatibility. Loading must happen before portable TORCH_LIBRARY schemas;
    deferring it to operator backend selection would duplicate definitions.
    """
    if "PPU_SDK" not in os.environ:
        return

    from .impl.native_extensions import load_all_native_extensions

    load_all_native_extensions()
