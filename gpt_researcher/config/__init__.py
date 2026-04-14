from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .config import Config
    from .variables.base import BaseConfig
    from .variables.default import DEFAULT_CONFIG as DefaultConfig


def __getattr__(name: str) -> Any:
    if name == "Config":
        from .config import Config as _Config

        return _Config
    if name == "BaseConfig":
        from .variables.base import BaseConfig as _BaseConfig

        return _BaseConfig
    if name == "DefaultConfig":
        from .variables.default import DEFAULT_CONFIG as _DefaultConfig

        return _DefaultConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Config", "BaseConfig", "DefaultConfig"]
