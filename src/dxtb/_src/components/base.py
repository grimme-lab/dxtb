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
Components: Base Class
======================

Base class for all tight-binding components.
"""

from __future__ import annotations

import torch

from dxtb.__version__ import __tversion__
from dxtb._src.typing import Any, Tensor, TensorLike
from dxtb._src.utils.misc import get_all_slots

__all__ = ["Component", "ComponentCache"]


class ComponentCache(TensorLike):
    """Cache of a component."""

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

    def __str__(self) -> str:
        slots = get_all_slots(self)
        s = ", ".join(s for s in slots if not s.startswith("_"))
        return f"{self.__class__.__name__}({s})"

    def __repr__(self) -> str:
        return str(self)


class Component(TensorLike):
    """
    Base class for all tight-binding terms.
    """

    label: str
    """Label for the tight-binding component."""

    _cache: ComponentCache | None
    """Cache for the component."""

    _cachevars: tuple[Tensor, ...] | None
    """
    Cache variable for the component.
    If this variable changes, the cache has to be rebuild.
    """

    _cache_enabled: bool
    """Flag to enable or disable the cache."""

    __slots__ = ["label", "_cache", "_cachevars", "_cache_enabled"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        _cache: ComponentCache | None = None,
        _cachevars: tuple[Tensor, ...] | None = None,
    ):
        super().__init__(device, dtype)
        self.label = self.__class__.__name__
        self._cache = _cache
        self._cachevars = _cachevars
        self._cache_enabled = True

    ############################################################################

    @property
    def cache(self) -> ComponentCache | None:
        """Cache for the interaction."""
        return self._cache

    @cache.setter
    def cache(self, value: ComponentCache | None) -> None:
        self._cache = value

    ############################################################################

    def update(self, **kwargs: Any) -> None:
        """
        Update the attributes of the :class:`.Component` instance.

        This method updates the attributes of the :class:`.Component`
        instance based on the provided keyword arguments. Only the attributes
        defined in ``__slots__`` can be updated.

        Parameters
        ----------
        kwargs : dict[str, Any]
            Keyword arguments where keys are attribute names and values are the
            new values for those attributes.
            Valid keys are those defined in ``__slots__`` of this class.

        Raises
        ------
        AttributeError
            If any key in kwargs is not an attribute defined in ``__slots__``.

        Examples
        --------

        .. code-block:: python

            import torch
            from dxtb._src.components.interactions.field import ElectricField

            ef = ElectricField(field=torch.tensor([0.0, 0.0, 0.0]))
            ef.update(field=torch.tensor([1.0, 0.0, 0.0]))
        """
        for key, value in kwargs.items():
            if key is None:
                continue

            if key in self.__slots__:
                setattr(self, key, value)
            else:
                raise AttributeError(
                    f"Cannot update '{key}' of the '{self.__class__.__name__}' "
                    "interaction. Invalid attribute."
                )

            self.cache_invalidate()

    def reset(self) -> None:
        """
        Reset the tensor attributes of the `Component` instance to their
        original states or to specified values.

        This method iterates through the attributes defined in ``__slots__`` and
        resets any tensor attributes to a detached clone of their original
        state. The `requires_grad` status of each tensor is preserved.

        Examples
        --------

        .. code-block:: python

            import torch
            from dxtb._src.components.interactions.external.field import ElectricField

            ef = ElectricField(field=torch.tensor([0.0, 0.0, 0.0]))
            ef.reset()

        Notes
        -----
        Only tensor attributes defined in ``__slots__`` are reset. Non-tensor
        attributes are ignored. Attempting to reset an attribute not defined in
        ``__slots__`` or providing a non-tensor value in `kwargs` will not raise
        an error; the method will simply ignore these cases and proceed with
        the reset operation for valid tensor attributes.
        """
        for slot in self.__slots__:
            attr = getattr(self, slot)
            if isinstance(attr, Tensor):
                reset = attr.detach().clone()
                reset.requires_grad = attr.requires_grad

                setattr(self, slot, reset)

        self.cache_invalidate()

    ############################################################################

    def cache_is_latest(
        self, vars: tuple[Tensor, ...], tol: float | None = None
    ) -> bool:
        """
        Check if the driver is set up and updated.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).

        Returns
        -------
        bool
            Flag for set up status.
        """
        if self._cache_enabled is False:
            return False

        if self.cache is None:
            return False

        if self._cachevars is None:
            return False

        for v1, v2 in zip(vars, self._cachevars):
            # functorch makes problems here, just disable cache for now
            if __tversion__ >= (1, 13, 0):
                if torch._C._functorch.is_gradtrackingtensor(v1):
                    return False
                if torch._C._functorch.is_gradtrackingtensor(v2):
                    return False

            if v1.dtype != v2.dtype:
                return False

            if v1.device != v2.device:
                return False

            if v1.dtype in (torch.int64, torch.int32, torch.long):
                if torch.equal(v1, v2) is False:
                    return False
            else:
                tol = torch.finfo(v1.dtype).eps ** 0.75 if tol is None else tol
                if (v2 - v1).abs().sum() > tol:
                    return False

        return True

    def cache_invalidate(self) -> None:
        """Invalidate the cache to require renewed setup."""
        self._cache = None
        self._cachevars = None

    @property
    def cache_is_setup(self) -> bool:
        """Whether the cache has been set up."""
        return self._cache is not None and self._cachevars is not None

    def cache_enable(self) -> None:
        """Enable the cache."""
        self._cache_enabled = True

    def cache_disable(self) -> None:
        """Disable the cache."""
        self._cache_enabled = False

    ############################################################################

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.label})"

    def __repr__(self) -> str:
        return str(self)
