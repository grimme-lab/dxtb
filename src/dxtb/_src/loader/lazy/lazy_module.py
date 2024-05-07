# This file is part of dxtb, modified from google/jax.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the Apache License, Version 2.0 by google/jax.
# Modifications made by Grimme Group.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Loader: Lazy Modules
====================

Contains the :func:`.attach_module` function that can be used to lazily load
submodules of a package.

Example
-------

.. code-block:: python

    from dxtb._src.loader.lazy import attach_module
    __getattr__, __dir__, __all__ = attach_module(__name__, ["sub1", "sub2"])
"""

from __future__ import annotations

import importlib

from dxtb._src.typing import Any, Callable, Sequence


def attach_module(package_name: str, submodules: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
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

    .. code-block:: python

        from dxtb._src.loader.lazy import attach_module
        __getattr__, __dir__, __all__ = attach_module(__name__, ["sub1", "sub2"])
    """

    __all__: list[str] = list(submodules)

    def __getattr__(name: str) -> Any:
        if name in submodules:
            return importlib.import_module(f"{package_name}.{name}")
        raise AttributeError(f"The module '{package_name}' has no attribute '{name}.")

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__
