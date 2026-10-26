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
Potential
=========

Container for collecting and handling the potentials within the SCF.

It currently supports the monopolar, dipolar and quadrupolar potential.
However, they are required in that very order, i.e., in increasing order of
multipole moment. That means a dipolar potential can not be used without a
monopolar potential.

For unrestricted (UHF) calculations, all properties carry an additional spin
dimension (charge/magnetization channels) before the orbital or atom
dimension, e.g., ``(..., 2, norb)`` or ``(..., 2, nat, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

import torch
from tad_mctc.batch import pack

from dxtb._src.constants import defaults
from dxtb._src.typing import ContainerData, Self, Tensor, Type, TypeVar

__all__ = ["Container", "ContainerLayout", "Charges", "Potential"]


T = TypeVar("T", bound="Container")


@dataclass
class ContainerLayout:
    """
    Shapes and metadata required to restore a :class:`.Container` from its
    tensor representation (see :meth:`.Container.as_tensor`).

    The shapes include the batch and spin dimensions. The number of spin
    channels is stored explicitly, i.e., the meaning of a dimension is never
    inferred from its size.
    """

    mono: torch.Size | None = None
    dipole: torch.Size | None = None
    quad: torch.Size | None = None
    label: list[str] = field(default_factory=list)
    batch_mode: int = 0
    nspin: int = 1

    def copy(self) -> ContainerLayout:
        """Return an independent copy of this layout."""
        return replace(self, label=self.label.copy())

    @property
    def component_shapes(self) -> list[torch.Size]:
        """Shapes of present components in monopole-to-quadrupole order."""
        shapes = [self.mono, self.dipole, self.quad]
        return [shape for shape in shapes if shape is not None]


def _swap_spin(shape: torch.Size, physical_ndim: int) -> list[int]:
    """Swap the spin dimension and the leading orbital/atom dimension."""
    swapped = list(shape)
    i = -(physical_ndim + 1)
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    return swapped


class Container:
    """
    Container for the density-dependent properties used in the SCF.
    """

    def __init__(
        self,
        mono: Tensor | None = None,
        dipole: Tensor | None = None,
        quad: Tensor | None = None,
        label: str | list[str] | None = None,
        batch_mode: int = 0,
        nspin: int = 1,
    ) -> None:
        self._mono = mono
        self._dipole = dipole
        self._quad = quad

        if label is not None:
            self.label = [label] if isinstance(label, str) else label
        else:
            self.label = []

        self.batch_mode = batch_mode
        self.axis = 1 if self.batch_mode else 0
        self.nspin = nspin

    @property
    def layout(self) -> ContainerLayout:
        """Layout required for restoring the container from a tensor."""
        return ContainerLayout(
            mono=self.mono_shape,
            dipole=self.dipole_shape,
            quad=self.quad_shape,
            label=self.label.copy(),
            batch_mode=self.batch_mode,
            nspin=self.nspin,
        )

    # monopole

    @property
    def mono(self) -> Tensor | None:
        return self._mono

    @mono.setter
    def mono(self, mono: Tensor) -> None:
        self._mono = mono

    @property
    def mono_shape(self) -> torch.Size | None:
        return self.mono.shape if self.mono is not None else None

    # dipole

    @property
    def dipole(self) -> Tensor | None:
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: Tensor) -> None:
        self._dipole = dipole

    @property
    def dipole_shape(self) -> torch.Size | None:
        return self.dipole.shape if self.dipole is not None else None

    # quadrupole

    @property
    def quad(self) -> Tensor | None:
        return self._quad

    @quad.setter
    def quad(self, quad: Tensor) -> None:
        self._quad = quad

    @property
    def quad_shape(self) -> torch.Size | None:
        return self.quad.shape if self.quad is not None else None

    def as_tensor(self, pad: int = defaults.PADNZ) -> Tensor:
        """
        Create a tensor representation of the container (stacking property).

        For two spin channels, the spin dimension is moved behind the leading
        orbital/atom dimension before flattening. Hence, truncating a
        flattened property (culling of converged systems in batched SCF)
        removes whole orbitals/atoms of both channels.

        Returns
        -------
        Tensor
            Stacked tensor of the property.

        Raises
        ------
        RuntimeError
            No tensors in the container class.
        """

        tensors = [
            t if self.nspin == 1 else t.movedim(-(ndim + 1), -ndim)
            for t, ndim in ((self.mono, 1), (self.dipole, 2), (self.quad, 2))
            if t is not None
        ]
        if not self.batch_mode:
            tensors = [t.flatten() for t in tensors]
        else:
            tensors = [t.flatten(start_dim=1) for t in tensors]

        if len(tensors) == 0:
            raise RuntimeError(
                "Container to tensor conversion requires at least one "
                "tensor. If no tensors should be used (empty), set the "
                "monopolar property to zero."
            )

        # only monopolar potential available (requires no packing but adding
        # the extra dimension must be done for consistent handling)
        if len(tensors) == 1:
            return tensors[0].unsqueeze(-2)

        # pack along dim=1 to keep the batch dimension in the first positions
        return pack(tensors, axis=self.axis, value=pad)

    def nullify_padding(self, pad: int = defaults.PADNZ) -> None:
        if self.mono is not None:
            zero = torch.tensor(
                0.0, device=self.mono.device, dtype=self.mono.dtype
            )
            self.mono = torch.where(self.mono != pad, self.mono, zero)

        if self.dipole is not None:
            zero = torch.tensor(
                0.0, device=self.dipole.device, dtype=self.dipole.dtype
            )
            self.dipole = torch.where(self.dipole != pad, self.dipole, zero)

        if self.quad is not None:
            zero = torch.tensor(
                0.0, device=self.quad.device, dtype=self.quad.dtype
            )
            self.quad = torch.where(self.quad != pad, self.quad, zero)

    @classmethod
    def from_tensor(
        cls: Type[T],
        tensor: Tensor,
        data: ContainerLayout | ContainerData,
        batch_mode: int = 0,
        pad: int = defaults.PADNZ,
    ) -> T:
        """
        Create a container from the tensor representation (stacked properties).
        This representation always assumes the following order within the
        stacked tensor: monopolar, dipolar, quadrupolar. Therefore, one cannot
        use a dipolar but no monopolar property.

        Parameters
        ----------
        tensor : Tensor
            Tensor representation of the container.
        data : ContainerLayout | ContainerData
            Collection of shapes and labels of the container. This information
            is required for correctly restoring the Container class.
        batch_mode : bool, optional
            Whether the calculation runs in batch_mode mode. Only used if
            `data` is not a :class:`ContainerLayout`, which already contains
            this information. Defaults to ``0``.
        pad : int, optional
            Value used to indicate padding. Not required, since the exact
            shapes are known. Defaults to ``defaults.PADNZ``.

        Returns
        -------
        Container
            Instance of the :class:`.Container` class.
        """
        if isinstance(data, ContainerLayout):
            layout = data
        else:
            label = data["label"]
            layout = ContainerLayout(
                mono=data["mono"],
                dipole=data["dipole"],
                quad=data["quad"],
                label=[label] if isinstance(label, str) else (label or []),
                batch_mode=batch_mode,
            )

        label = layout.label.copy()
        batch_mode = layout.batch_mode
        nspin = layout.nspin

        # no property dimension, i.e., only monopolar property
        ndim = 2 if batch_mode else 1
        if tensor.shape == layout.mono or (nspin == 1 and tensor.ndim == ndim):
            return cls(
                mono=tensor, label=label, batch_mode=batch_mode, nspin=nspin
            )

        axis = 1 if batch_mode else 0
        components: list[Tensor] = []
        for i, (shape, physical_ndim) in enumerate(
            zip(layout.component_shapes, (1, 2, 2))
        ):
            if i >= tensor.shape[axis]:
                break

            prop = tensor.select(axis, i)
            size = prod(shape[1:]) if batch_mode else prod(shape)
            prop = prop[..., :size]

            if nspin == 1:
                components.append(prop.reshape(shape))
            else:
                # undo spin-inner order of `as_tensor`
                components.append(
                    prop.reshape(_swap_spin(shape, physical_ndim)).movedim(
                        -physical_ndim, -(physical_ndim + 1)
                    )
                )

        values: list[Tensor | None] = [*components, None, None]
        return cls(
            mono=values[0],
            dipole=values[1],
            quad=values[2],
            label=label,
            batch_mode=batch_mode,
            nspin=nspin,
        )

    def select_channel(self, channel: int) -> Self:
        """
        Select the charge (``0``) or magnetization (``1``) channel.

        Parameters
        ----------
        channel : int
            Index of the channel.

        Returns
        -------
        Self
            One-channel container.
        """
        if self.nspin == 1:
            if channel != 0:
                raise RuntimeError(
                    "The magnetization channel requires two spin channels."
                )
            return self

        def select(t: Tensor | None, physical_ndim: int) -> Tensor | None:
            return (
                None if t is None else t.select(-(physical_ndim + 1), channel)
            )

        return self.__class__(
            mono=select(self.mono, 1),
            dipole=select(self.dipole, 2),
            quad=select(self.quad, 2),
            label=self.label.copy(),
            batch_mode=self.batch_mode,
        )

    def to_spin_channels(self, channel: int) -> Self:
        """
        Embed a one-channel container into the charge (``0``) or
        magnetization (``1``) channel of a two-channel container.

        Parameters
        ----------
        channel : int
            Index of the channel.

        Returns
        -------
        Self
            Two-channel container. The other channel is zero.
        """

        def embed(t: Tensor | None, physical_ndim: int) -> Tensor | None:
            if t is None:
                return None
            zero = torch.zeros_like(t)
            channels = (t, zero) if channel == 0 else (zero, t)
            return torch.stack(channels, dim=-(physical_ndim + 1))

        return self.__class__(
            mono=embed(self.mono, 1),
            dipole=embed(self.dipole, 2),
            quad=embed(self.quad, 2),
            label=self.label.copy(),
            batch_mode=self.batch_mode,
            nspin=2,
        )

    def add_tensors(
        self, tensor1: Tensor | None, tensor2: Tensor | None
    ) -> Tensor | None:
        """
        Add two tensors together, while handling ``None``.

        Parameters
        ----------
        tensor1 : Tensor | None
            First tensor.
        tensor2 : Tensor | None
            Second tensor.

        Returns
        -------
        Tensor | None
            Added tensors or ``None`` if both tensor are ``None``.
        """
        if tensor1 is None:
            return tensor2
        if tensor2 is None:
            return tensor1
        return tensor1 + tensor2

    def _check_addition(self, other: Container) -> None:
        if not isinstance(other, Container):
            raise TypeError("Only the same containers can be added together")

        if other.nspin != self.nspin:
            raise ValueError(
                "Only containers with the same number of spin channels can be "
                f"added together ({self.nspin} != {other.nspin})."
            )

        if other.label in self.label:
            raise ValueError(
                f"A property with the label '{other.label}' already exists in "
                "the Container you are adding to."
            )

    def __add__(self, other: Container) -> Container:
        self._check_addition(other)

        return self.__class__(
            mono=self.add_tensors(self.mono, other.mono),
            dipole=self.add_tensors(self.dipole, other.dipole),
            quad=self.add_tensors(self.quad, other.quad),
            label=self.label + other.label,
            batch_mode=self.batch_mode,
            nspin=self.nspin,
        )

    def __iadd__(self, other: Container) -> Self:
        self._check_addition(other)

        self._mono = self.add_tensors(self.mono, other.mono)
        self._dipole = self.add_tensors(self.dipole, other.dipole)
        self._quad = self.add_tensors(self.quad, other.quad)
        self.label += other.label

        return self

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"{self.__class__.__name__}("
            f"label={self.label!r}, "
            f"mono={self.mono!r}, "
            f"dipole={self.dipole!r}, "
            f"quad={self.quad!r}, "
            f"batch_mode={self.batch_mode!r}, "
            f"nspin={self.nspin!r})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


class Charges(Container):
    """
    Container for the charges used in the SCF.
    """

    @property
    def mono(self) -> Tensor:
        if self._mono is None:
            raise RuntimeError("Monopole charges are always required.")
        return self._mono

    @mono.setter
    def mono(self, mono: Tensor) -> None:
        self._mono = mono

    def __str__(self) -> str:  # pragma: no cover
        dp_shape = self.dipole.shape if self.dipole is not None else None
        qp_shape = self.quad.shape if self.quad is not None else None
        return (
            f"{self.__class__.__name__}("
            f"mono={self.mono.shape!r}, "
            f"dipole={dp_shape!r}, "
            f"quad={qp_shape!r}, "
            f"batch_mode={self.batch_mode!r}, "
            f"nspin={self.nspin!r})"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


class Potential(Container):
    """
    Container for the density-dependent potentials used in the SCF.
    """

    def reset(self) -> None:
        if self.mono is not None:
            self.mono = torch.zeros_like(self.mono)

        if self.dipole is not None:
            self.dipole = torch.zeros_like(self.dipole)

        if self.quad is not None:
            self.quad = torch.zeros_like(self.quad)
