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
Components: List
================

Base class for all collections (lists) of tight-binding components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic

import torch

from dxtb.typing import Any, Self, Slicers, Tensor, TensorLike, TypeVar, override

from .base import Component

C = TypeVar("C", bound=Component)


class ComponentListABC(ABC):
    """
    Abstract base class for component lists.
    """

    class Cache(dict):
        """
        List of component caches.
        """

    @abstractmethod
    def get_energy(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute the energy for a list of components.
        """

    @abstractmethod
    def get_gradient(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        Compute the gradient for a list of components.
        """

    @abstractmethod
    def get_cache(self, *args: Any, **kwargs: Any) -> Cache:
        """
        Create restart data for all components.
        """


class ComponentList(ComponentListABC, Generic[C], TensorLike):
    """
    List of tight-binding components.
    """

    components: list[C]
    """List of tight-binding components instances."""

    __slots__ = ("components",)

    class Cache(ComponentListABC.Cache):
        """
        List of component caches.

        Note
        ----
        The cache class inherits from `dict`.
        """

        __slots__ = ()

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            """
            Cull all interaction caches.

            Parameters
            ----------
            conv : Tensor
                Mask of converged systems.
            """
            for cache in self.values():
                cache.cull(conv, slicers)

        def restore(self) -> None:
            """
            Restore all interaction caches.
            """
            for cache in self.values():
                cache.restore()

    def __init__(
        self,
        *components: C | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Instantiate a collection of tight-binding components.

        Parameters
        ----------
        compontents : tuple[Component | None, ...] | list[Component | None]
            List or tuple of compontent classes.

        Note
        ----
        Duplicate components will be removed automatically.
        """
        # FIXME: I doubt that moving a list of classes works in TensorLike...
        super().__init__(device, dtype)
        self.components = list({c for c in components if c is not None})

    def update(self, name: str, **kwargs: Any) -> C:
        """
        Update the attributes of an interaction object within the list.

        This method iterates through the interactions in the list, finds the
        one with the matching label, and updates its attributes based on the
        provided arguments.

        Parameters
        ----------
        name : str
            The label of the interaction object to be updated.
        **kwargs : dict
            Keyword arguments containing the attributes and their new values to
            be updated in the interaction object.

        Raises
        ------
        ValueError
            If no interaction with the given label is found in the list.

        Examples
        --------
        >>> from dxtb.components.interactions import InteractionList
        >>> from dxtb.components.interactions.external import field as efield
        >>>
        >>> field_vector = torch.tensor([0.0, 0.0, 0.0])
        >>> new_field_vector = torch.tensor([1.0, 0.0, 0.0])
        >>> ef = efield.new_efield(field_vector)
        >>> ilist = InteractionList(ef)
        >>> ilist.update(efield.LABEL_EFIELD, field=new_field_vector)
        """
        for interaction in self.components:
            if name == interaction.label:
                interaction.update(**kwargs)
                return interaction

        raise ValueError(f"The interaction named '{name}' is not in the list.")

    def reset(self, name: str) -> C:
        """
        Reset the attributes of an component object within the list.

        This method iterates through the components in the list, finds the
        one with the matching label, and resets any tensor attributes to a
        detached clone of their original state. The `requires_grad` status of
        each tensor is preserved.

        Parameters
        ----------
        name : str
            The label of the component object to be updated.

        Returns
        -------
        Component
            The component object with the resetted attributes.

        Raises
        ------
        ValueError
            If no component with the given label is found in the list.

        Examples
        --------
        >>> import torch
        >>> from dxtb.components.interactions import InteractionList
        >>> from dxtb.components.interactions.external import field as efield
        >>>
        >>> ef = efield.new_efield(torch.tensor([0.0, 0.0, 0.0]))
        >>> ilist = InteractionList(ef)
        >>> ilist.reset(efield.LABEL_EFIELD)
        """
        for component in self.components:
            if name == component.label:
                component.reset()
                return component

        raise ValueError(f"The component named '{name}' is not in the list.")

    def reset_all(self) -> None:
        """Reset tensor attributes to a detached clone of the current state."""
        for component in self.components:
            component.reset()

    @property
    def labels(self) -> list[str]:
        return [component.label for component in self.components]

    def get_interaction(self, name: str) -> C:
        """
        Obtain an component from the list of components by its class name.

        Parameters
        ----------
        name : str
            Name of the component.

        Returns
        -------
        Interaction
            Instance of the component as present in the `InteractionList`.

        Raises
        ------
        ValueError
            Unknown component name given or component is not in the list.
        """
        for component in self.components:
            if name == component.label:
                return component

        raise ValueError(f"The component named '{name}' is not in the list.")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.labels})"

    def __repr__(self) -> str:
        return str(self)

    @override
    def type(self, dtype: torch.dtype) -> Self:
        self.components = list({c.type(dtype) for c in self.components})
        self.override_dtype(dtype)
        return self


def _docstring_update(func):
    """
    Decorator to assign a generic docstring to update methods.
    The docstring is generated based on the method name.
    """
    attribute_name = func.__name__.replace("update_", "")

    docstring = f"""
    Update the attribute of the '{attribute_name}' object based on the provided
    arguments in the keyword arguments.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments containing the attributes and their new values to
        be updated in the component object.

    Returns
    -------
    Component
        The component object with the updated attributes.

    Raises
    ------
    ValueError
        If no component with the given label is found in the list.

    Examples
    --------
    See `ComponentList.update`.
    """
    func.__doc__ = docstring
    return func


def _docstring_reset(func):
    """
    Decorator to assign a generic docstring to update methods.
    The docstring is generated based on the method name.
    """
    attribute_name = func.__name__.replace("update_", "")

    docstring = f"""
    Reset the attributes of the '{attribute_name}' object within the list.

    This method resets any tensor attributes to a detached clone of their
    original state. The `requires_grad` status of each tensor is preserved.

    Returns
    -------
    Component
        The component object with the resetted attributes.

    Raises
    ------
    ValueError
        If no component with the given label is found in the list.
    """
    func.__doc__ = docstring
    return func
