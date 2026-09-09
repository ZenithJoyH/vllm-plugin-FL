# Copyright (c) 2026 BAAI. All rights reserved.

"""T-Head backend exports, kept lazy for early native-schema initialization."""

__all__ = ["TheadBackend"]


def __getattr__(name: str):
    if name == "TheadBackend":
        from .thead import TheadBackend

        globals()[name] = TheadBackend
        return TheadBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
