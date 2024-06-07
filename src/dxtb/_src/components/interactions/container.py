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
"""

from __future__ import annotations

import torch
from tad_mctc.batch import deflate, pack

from dxtb._src.constants import defaults
from dxtb._src.typing import ContainerData, Self, Tensor, Type, TypeVar

__all__ = ["Container", "Charges", "Potential"]


T = TypeVar("T", bound="Container")


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

        Returns
        -------
        Tensor
            Stacked tensor of the property.

        Raises
        ------
        RuntimeError
            No tensors in the container class.
        """

        tensors = [self.mono, self.dipole, self.quad]
        if not self.batch_mode:
            tensors = [t.flatten() for t in tensors if t is not None]
        else:
            tensors = [t.flatten(start_dim=1) for t in tensors if t is not None]

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
            zero = torch.tensor(0.0, device=self.mono.device, dtype=self.mono.dtype)
            self.mono = torch.where(self.mono != pad, self.mono, zero)

        if self.dipole is not None:
            zero = torch.tensor(0.0, device=self.dipole.device, dtype=self.dipole.dtype)
            self.dipole = torch.where(self.dipole != pad, self.dipole, zero)

        if self.quad is not None:
            zero = torch.tensor(0.0, device=self.quad.device, dtype=self.quad.dtype)
            self.quad = torch.where(self.quad != pad, self.quad, zero)

    @classmethod
    def from_tensor(
        cls: Type[T],
        tensor: Tensor,
        data: ContainerData,
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
        data : ContainerData
            Collection of shapes and labels of the container. This information
            is required for correctly restoring the the Container class.
        batch_mode : bool, optional
            Whether the calculation runs in batch_mode mode. Defaults to ``False``.
        pad : int, optional
            Value used to indicate padding. Defaults to ``defaults.PADNZ``.

        Returns
        -------
        Container
            Instance of the `Container` class.
        """

        ndim = tensor.ndim
        label = data["label"]
        axis = 1 if batch_mode else 0

        if (ndim == 1 and not batch_mode) or (ndim == 2 and batch_mode):
            return cls(mono=tensor, label=label)

        # One dimensions extra for more than monopole ...
        if (ndim == 2 and not batch_mode) or (ndim == 3 and batch_mode):
            # ... but still account for (nb, 1, nao)-shaped monopolar property.
            assert data["mono"] is not None
            if tensor.shape[axis] == 1:
                return cls(mono=tensor.reshape(*data["mono"]), label=label)

            # Now, dipolar and quadrupolar properties are checked.
            assert data["dipole"] is not None
            vs = torch.split(tensor, 1, dim=axis)

            # TODO: Conformer batch_mode mode (deflate not required)
            mono = deflate(vs[0], axis=0, value=pad).reshape(*data["mono"])
            dipole = deflate(vs[1], axis=0, value=pad).reshape(*data["dipole"])

            if tensor.shape[axis] == 2:
                return cls(mono, dipole, label=label)

            assert data["quad"] is not None
            quad = deflate(vs[2], axis=0, value=pad).reshape(*data["quad"])
            if tensor.shape[axis] == 3:
                return cls(mono, dipole, quad, label=label)

            raise RuntimeError(
                "It appears as if more than 3 tensors are given in the "
                f"tensor representation as its shape is {tensor.shape}."
            )

        raise RuntimeError(
            f"The tensor representation has {tensor.ndim} dimension but "
            "should have 2 (non-batch_mode) or 3 (batch_mode)."
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

    def __add__(self, other: Container) -> Container:
        if not isinstance(other, Container):
            raise TypeError("Only the same containers can be added together")

        if other.label in self.label:
            raise ValueError(
                f"A property with the label '{other.label}' already exists in "
                "the Container you are adding to."
            )

        return Container(
            mono=self.add_tensors(self.mono, other.mono),
            dipole=self.add_tensors(self.dipole, other.dipole),
            quad=self.add_tensors(self.quad, other.quad),
            label=self.label + other.label,
        )

    def __iadd__(self, other: Container) -> Self:
        if not isinstance(other, Container):
            raise TypeError("Only the same containers can be added together")

        if other.label in self.label:
            raise ValueError(
                f"A property with the label '{other.label}' already exists in "
                "the Container you are adding to."
            )

        self._mono = self.add_tensors(self.mono, other.mono)
        self._dipole = self.add_tensors(self.dipole, other.dipole)
        self._quad = self.add_tensors(self.quad, other.quad)
        self.label += other.label

        return self

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"label={self.label!r}, "
            f"mono={self.mono!r}, "
            f"dipole={self.dipole!r}, "
            f"quad={self.quad!r}, "
            f"batch_mode={self.batch_mode!r})"
        )


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

    def __repr__(self):
        dp_shape = self.dipole.shape if self.dipole is not None else None
        qp_shape = self.quad.shape if self.quad is not None else None
        return (
            f"{self.__class__.__name__}("
            f"mono={self.mono.shape!r}, "
            f"dipole={dp_shape!r}, "
            f"quad={qp_shape!r}, "
            f"batch_mode={self.batch_mode!r})"
        )


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
