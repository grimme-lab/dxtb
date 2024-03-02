"""
A LazyLoader class for loading.
"""

import importlib

from tad_mctc.typing import Any, Callable, Sequence


def attach_module(
    package_name: str, submodules: Sequence[str]
) -> tuple[Callable[[str], Any], Callable[[], list[str]], list[str],]:
    """
    Lazily loads submodules of a given package, providing a way to access them
    on demand.

    This function is intended to be used in a package's `__init__.py` file to
    allow lazy loading of its submodules.
    It returns a tuple containing two callables (`__getattr__` and `__dir__`)
    and a list of submodule names (`__all__`).
    `__getattr__` is used to load a submodule when it's accessed for the first
    time, while `__dir__` lists the available submodules.

    Parameters
    ----------
    package_name : str
        The name of the package for which submodules are to be lazily loaded.
    submodules : Sequence[str]
        A sequence of strings representing the names of the submodules to be
        lazily loaded.

    Returns
    -------
    tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]
        A tuple containing:
        - A `__getattr__` function loading a submodule when it's accessed.
        - A `__dir__` function returning a list of all lazily loaded submodules.
        - A list of strings (`__all__`) containing the names of the submodules.

    Raises
    ------
    AttributeError
        Raised when an attempt is made to access a submodule that is not listed
        in the `submodules` parameter.

    Example
    -------
    >>> __getattr__, __dir__, __all__ = attach_module(__name__, ["sub1", "sub2"])
    """

    __all__: list[str] = list(submodules)

    def __getattr__(name: str) -> Any:
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        raise AttributeError(f"The module '{package_name}' has no attribute '{name}.")

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__
