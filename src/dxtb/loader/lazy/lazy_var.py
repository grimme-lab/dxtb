"""
A LazyLoader class for loading variables.
"""

from __future__ import annotations

import importlib

from tad_mctc.typing import Any, Callable, Sequence


def attach_var(package_name: str, varnames: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
    __all__: list[str] = list(varnames)

    def __getattr__(name: str) -> Any:
        if name not in varnames:
            raise AttributeError(
                f"The module '{package_name}' has no attribute '{name}."
            )

        module = importlib.import_module(f"{package_name}")

        return getattr(module, name)

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__
