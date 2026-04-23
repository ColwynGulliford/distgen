from __future__ import annotations

import typing as _typing

if _typing.TYPE_CHECKING:
    from .generator import Generator
    from .physical_constants import PHYSICAL_CONSTANTS

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = ["Generator", "PHYSICAL_CONSTANTS"]


def __dir__():
    return __all__


def __getattr__(name: str):
    if name == "Generator":
        from .generator import Generator

        globals()["Generator"] = Generator
        return Generator
    if name == "PHYSICAL_CONSTANTS":
        from .physical_constants import PHYSICAL_CONSTANTS

        globals()["PHYSICAL_CONSTANTS"] = PHYSICAL_CONSTANTS
        return PHYSICAL_CONSTANTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
