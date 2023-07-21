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

from .._types import PotentialData, Self, Tensor
from ..constants import defaults
from ..utils.batch import deflate, pack


class Potential:
    """
    Container for the density-dependent potentials used in the SCF.
    """

    def __init__(
        self,
        vmono: Tensor | None = None,
        vdipole: Tensor | None = None,
        vquad: Tensor | None = None,
        label: str | list[str] | None = None,
        batched: bool = False,
    ) -> None:
        self._vmono = vmono
        self._vdipole = vdipole
        self._vquad = vquad

        if label is not None:
            self.label = [label] if isinstance(label, str) else label
        else:
            self.label = []

        self.batched = batched

    # monopole

    @property
    def vmono(self) -> Tensor | None:
        return self._vmono

    @vmono.setter
    def vmono(self, vmono: Tensor) -> None:
        self._vmono = vmono

    @property
    def vmono_shape(self) -> torch.Size | None:
        return self.vmono.shape if self.vmono is not None else None

    # dipole

    @property
    def vdipole(self) -> Tensor | None:
        return self._vdipole

    @vdipole.setter
    def vdipole(self, vdipole: Tensor) -> None:
        self._vdipole = vdipole

    @property
    def vdipole_shape(self) -> torch.Size | None:
        return self.vdipole.shape if self.vdipole is not None else None

    # quadrupole

    @property
    def vquad(self) -> Tensor | None:
        return self._vquad

    @vquad.setter
    def vquad(self, vquad: Tensor) -> None:
        self._vquad = vquad

    @property
    def vquad_shape(self) -> torch.Size | None:
        return self.vquad.shape if self.vquad is not None else None

    # functionality

    def reset(self) -> None:
        if self.vmono is not None:
            self.vmono = torch.zeros_like(self.vmono)

        if self.vdipole is not None:
            self.vdipole = torch.zeros_like(self.vdipole)

        if self.vquad is not None:
            self.vquad = torch.zeros_like(self.vquad)

    def as_tensor(self, pad: int = defaults.PADNZ) -> Tensor:
        """
        Create a tensor representation of the potential (stacking potentials).

        Returns
        -------
        Tensor
            Stacked tensor of potentials.

        Raises
        ------
        RuntimeError
            No tensors in the potential class.
        """

        tensors = [self.vmono, self.vdipole, self.vquad]
        if not self.batched:
            tensors = [t.flatten() for t in tensors if t is not None]
        else:
            tensors = [t.flatten(start_dim=1) for t in tensors if t is not None]

        if len(tensors) == 0:
            raise RuntimeError(
                "Potential to tensor conversion requires at least one "
                "potential. If no potential should be used, set the monopolar "
                "potential to zero."
            )

        # only monopolar potential available
        if len(tensors) == 1:
            return tensors[0]

        return pack(tensors, value=pad)

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor,
        data: PotentialData,
        batched: bool = False,
        pad: int = defaults.PADNZ,
    ) -> Potential:
        """
        Create a potential from the tensor representation (stacked potentials).
        This representation always assumes the following order within the
        stacked tensor: monopolar, dipolar, quadrupolar. Therefore, one cannot
        use a dipolar but no monopolar potential.

        Parameters
        ----------
        tensor : Tensor
            Tensor representation of the potentials.
        data : PotentialData
            Collection of shapes and labels of the potential. This information
            is required for correctly restoring the the Potential.
        batched : bool, optional
            Whether the calculation runs in batched mode. Defaults to `False`.
        pad : int, optional
            Value used to indicate padding. Defaults to ``defaults.PADNZ``.

        Returns
        -------
        Potential
            Instance of the `Potential` class.
        """
        ndim = tensor.ndim
        label = data["label"]

        if (ndim == 1 and not batched) or (ndim == 2 and batched):
            return cls(vmono=tensor, label=label)

        # One dimensions extra for more than monopole ...
        if (ndim == 2 and not batched) or (ndim == 3 and batched):
            # ... but still account for (1, nb, nao)-shaped monopolar potential.
            if tensor.shape[0] == 1:
                return cls(vmono=tensor.reshape(*data["mono"]), label=label)

            # Now, dipolar and quadrupolar potentials are checked.
            vs = torch.split(tensor, 1, dim=0)
            vmono = vs[0].reshape(*data["mono"])
            vdipole = deflate(vs[1], value=pad).reshape(*data["dipole"])
            if tensor.shape[0] == 2:
                return cls(vmono, vdipole, label=label)

            vquad = deflate(vs[2], value=pad).reshape(*data["quad"])
            if tensor.shape[0] == 3:
                return cls(vmono, vdipole, vquad, label=label)

            raise RuntimeError(
                "It appears as if more than 3 potentials are given in the "
                f"tensor representation as its shape is {tensor.shape}."
            )

        raise RuntimeError(
            f"The tensor representation has {tensor.ndim} dimension but "
            "should have 2 (non-batched) or 3 (batched)."
        )

    def add_tensors(
        self, tensor1: Tensor | None, tensor2: Tensor | None
    ) -> Tensor | None:
        """
        Add to tensors together, while handling `None`s.

        Parameters
        ----------
        tensor1 : Tensor | None
            First tensor.
        tensor2 : Tensor | None
            Second tensor.

        Returns
        -------
        Tensor | None
            Added tensors or `None` if both tensor are `None`.
        """
        if tensor1 is None:
            return tensor2
        if tensor2 is None:
            return tensor1
        return tensor1 + tensor2

    def __add__(self, other: Potential) -> Potential:
        if not isinstance(other, Potential):
            raise TypeError("Only potentials can be added together")

        if other.label in self.label:
            raise ValueError(
                f"A potential with the label '{other.label}' already exists in "
                "the Potential you are adding to."
            )

        return Potential(
            vmono=self.add_tensors(self.vmono, other.vmono),
            vdipole=self.add_tensors(self.vdipole, other.vdipole),
            vquad=self.add_tensors(self.vquad, other.vquad),
            label=self.label + other.label,
        )

    def __iadd__(self, other: Potential) -> Self:
        if not isinstance(other, Potential):
            raise TypeError("Only potentials can be added together")

        if other.label in self.label:
            raise ValueError(
                f"A potential with the label '{other.label}' already exists in "
                "the Potential you are adding to."
            )

        self._vmono = self.add_tensors(self.vmono, other.vmono)
        self._vdipole = self.add_tensors(self.vdipole, other.vdipole)
        self._vquad = self.add_tensors(self.vquad, other.vquad)
        self.label += other.label

        return self

    def __repr__(self):
        return (
            f"Potential(label={self.label!r}, "
            f"vmono={self.vmono!r}, "
            f"vdipole={self.vdipole!r}, "
            f"vquad={self.vquad!r})"
        )
