from __future__ import annotations

from typing import Any

__all__ = ["HomelabTUI"]


def __getattr__(name: str) -> Any:
    if name == "HomelabTUI":
        from .app import HomelabTUI

        return HomelabTUI
    raise AttributeError(name)
