from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _inject_workspace_vendor_paths() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    llm_fallbacks_src = repo_root.parent / "llm_fallbacks" / "src"

    if not llm_fallbacks_src.is_dir():
        return

    llm_fallbacks_str = str(llm_fallbacks_src)
    if llm_fallbacks_str not in sys.path:
        sys.path.insert(0, llm_fallbacks_str)


_inject_workspace_vendor_paths()

if TYPE_CHECKING:
    from .agent import GPTResearcher


def __getattr__(name: str) -> Any:
    if name == "GPTResearcher":
        from .agent import GPTResearcher as _GPTResearcher

        return _GPTResearcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["GPTResearcher"]