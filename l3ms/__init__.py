from __future__ import annotations

from typing import Any

__all__ = ["L3MSApp"]


def __getattr__(name: str) -> Any:
    if name == "L3MSApp":
        from .app import L3MSApp

        return L3MSApp
    raise AttributeError(name)
