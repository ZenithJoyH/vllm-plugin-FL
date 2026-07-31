"""Copyright (c) 2026 BAAI. All rights reserved."""

# Import attention module for FA3 support (side-effect: patches flash_attn)
from .attention import TheadFlashAttentionBackend, TheadFlashAttentionImpl

# Import mla module for FlashMLA support (side-effect: patches flashmla ops)
# Must come after attention import since they share the flash_attn namespace
from . import mla  # noqa: F401

__all__ = ["TheadFlashAttentionBackend", "TheadFlashAttentionImpl"]
