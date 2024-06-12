# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
Loaders: Lazy Parameter Loader
==============================

A lazy loader class for loading TOML parametrization upon member access.
This is used for loading the GFN1-xTB and GFN2-xTB parametrizations.

Example
-------
.. code-block:: python

    from dxtb._src.loader.lazy import LazyLoaderParam
    param = LazyLoaderParam("../../param/gfn1/gfn1-xtb.toml")
"""

from __future__ import annotations

from dxtb._src.typing import Any, PathLike

__all__ = ["LazyLoaderParam"]


class LazyLoaderParam:
    """
    A lazy loader class for loading TOML parametrization files as needed.

    This class is designed to delay the loading of a TOML file until an
    attribute from the file is accessed. It dynamically loads and parses the
    TOML file, initializing a `Param` object with the parsed data only upon
    attribute access.

    Parameters
    ----------
    filepath : PathLike
        The file path to the TOML file that needs to be lazily loaded.

    Attributes
    ----------
    filepath : PathLike
        Stores the file path of the TOML file.
    _loaded : Param or None
        Stores the loaded `Param` object after the first attribute access.

    Methods
    -------
    __getattr__(item: Any) -> Any
        Overridden method to load the TOML file and access attributes of the
        `Param` object.
    """

    def __init__(self, filepath: PathLike) -> None:
        """
        Initializes the LazyLoaderParam with the specified file path.

        Parameters
        ----------
        filepath : PathLike
            The file path to the TOML file that needs to be lazily loaded.
        """
        self.filepath = filepath
        self._loaded = None

    def __getattr__(self, item: Any) -> Any:
        """
        Loads the TOML file and initializes the `Param` object upon first
        attribute access.
        Subsequent accesses will use the already loaded `Param` object.

        Parameters
        ----------
        item : Any
            The attribute name to be accessed from the `Param` object.

        Returns
        -------
        Any
            The value of the attribute from the `Param` object.
        """
        if self._loaded is None:
            import tomli as toml

            from dxtb._src.param.base import Param

            with open(self.filepath, "rb") as fd:
                self._loaded = Param(**toml.load(fd))

            del toml

        return getattr(self._loaded, item)

    def __str__(self) -> str:
        """
        Custom string representation of the `LazyLoaderParam` object.

        Returns
        -------
        str
            The string representation of the `LazyLoaderParam` object.
        """
        return f"LazyLoaderParam({str(self.filepath)})"

    def __repr__(self) -> str:
        """
        Custom representation of the `LazyLoaderParam` object.

        Returns
        -------
        str
            The string representation of the `LazyLoaderParam` object.
        """
        return str(self)

    def __eq__(self, other: Any) -> bool:
        """
        Check if the :class:`.LazyLoaderParam` object is equal to another
        object.

        Parameters
        ----------
        value : object
            The object to compare with the :class:`.LazyLoaderParam` object.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        return self._loaded == other
