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
Loaders: Lazy Parameter Variables
=================================

Contains the `:func:attach_var` function that can be used to lazily load
variables from submodules of a package.

Example
-------

.. code-block:: python

    from dxtb._src.loader.lazy import attach_var
    __getattr__, __dir__, __all__ = attach_var("dxtb.mol.molecule", ["Mol"])

Multiple variables can be lazily loaded using `:func:attach_vars`.

.. code-block:: python

    from dxtb._src.loader.lazy import attach_vars
    __getattr__, __dir__, __all__ = attach_vars(
        {
            "dxtb.properties.moments.dipole": ["dipole"],
            "dxtb.properties.moments.quadrupole": ["quadrupole"],
        }
    )
"""

from __future__ import annotations

import importlib

from dxtb._src.typing import Any, Callable, Sequence

__all__ = ["attach_var", "attach_vars"]


def attach_var(package_name: str, varnames: Sequence[str]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
    """
    Lazily loads variables from submodules of a given package, providing a way
    to access them on demand.

    This function is intended to be used in a package's `__init__.py` file to
    allow lazy loading of its variables.
    It returns a tuple containing two callables (`__getattr__` and `__dir__`)
    and a list of variable names (`__all__`).
    `__getattr__` is used to load a variable when it's accessed for the first
    time, while `__dir__` lists the available variables.

    Parameters
    ----------
    package_name : str
        The name of the package for which variables are to be lazily loaded.
    varnames : Sequence[str]
        A sequence of variable names to be lazily loaded.

    Returns
    -------
    tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]
        A tuple containing:

        - A `__getattr__` function loading a variable when it's accessed.
        - A `__dir__` function returning a list of all lazily loaded variable.
        - A list of strings (`__all__`) containing the names of the variable.

    Raises
    ------
    AttributeError
        If the package does not have the requested variable.

    Example
    -------

    .. code-block:: python

        from dxtb._src.loader.lazy import attach_var
        __getattr__, __dir__, __all__ = attach_var("dxtb.mol.molecule", ["Mol"])
    """
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


def attach_vars(module_vars: dict[str, Sequence[str]]) -> tuple[
    Callable[[str], Any],
    Callable[[], list[str]],
    list[str],
]:
    """
    Lazily loads variables from multiple submodules of a given package,
    providing a way to access them on demand.

    Parameters
    ----------
    module_vars : dict[str, Sequence[str]]
        A dictionary mapping package names to sequences of variable names to be
        lazily loaded.

    Returns
    -------
    tuple[Callable[[str], Any], Callable[[], list[str]], list[str]]
        A tuple containing:

        - A `__getattr__` function loading a variable when it's accessed.
        - A `__dir__` function returning a list of all lazily loaded variable.
        - A list of strings (`__all__`) containing the names of the variable.

    Raises
    ------
    AttributeError
        If the package does not have the requested variable.

    Example
    -------

    .. code-block:: python

        from dxtb._src.loader.lazy import attach_vars
        __getattr__, __dir__, __all__ = attach_vars(
            {
                "dxtb.properties.moments.dipole": ["dipole"],
                "dxtb.properties.moments.quadrupole": ["quadrupole"],
            }
        )
    """
    # Aggregate all variable names into __all__
    __all__: list[str] = [var for vars in module_vars.values() for var in vars]

    def __getattr__(name: str) -> Any:
        for package_name, varnames in module_vars.items():
            if name in varnames:
                module = importlib.import_module(package_name)
                return getattr(module, name)
        raise AttributeError(f"No module contains the variable '{name}'.")

    def __dir__() -> list[str]:
        return __all__

    return __getattr__, __dir__, __all__
