"""
External Fields
===============

Interaction of the charge density with external fields.
"""
from __future__ import annotations

import torch

from dxtb._types import Tensor

from ..._types import Slicers, Tensor, TensorLike
from ...constants import defaults
from ...exceptions import DeviceError, DtypeError
from ..base import Interaction
from ..container import Charges

__all__ = ["ElectricField", "LABEL_EFIELD", "new_efield"]


LABEL_EFIELD = "ElectricField"
"""Label for the 'ElectricField' interaction, coinciding with the class name."""


class ElectricField(Interaction):
    """
    Instantaneous electric field.
    """

    field: Tensor
    """Instantaneous electric field vector."""

    field_grad: Tensor | None
    """Electric field gradient."""

    __slots__ = ["field", "field_grad"]

    class Cache(Interaction.Cache, TensorLike):
        """
        Restart data for the electric field interaction.
        """

        __store: Store | None
        """Storage for cache (required for culling)."""

        vat: Tensor
        """
        Atom-resolved monopolar potental from instantaneous electric field.
        """

        vdp: Tensor
        """
        Atom-resolved dipolar potential from instantaneous electric field.
        """

        vqp: Tensor | None
        """
        Atom-resolved quadrupolar potential from electric field gradient.
        """

        __slots__ = ["__store", "vat", "vdp", "vqp"]

        def __init__(
            self,
            vat: Tensor,
            vdp: Tensor,
            vqp: Tensor | None = None,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else vat.device,
                dtype=dtype if dtype is None else vat.dtype,
            )
            self.vat = vat
            self.vdp = vdp
            self.vqp = vqp
            self.__store = None

        class Store:
            """
            Storage container for cache containing `__slots__` before culling.
            """

            vat: Tensor
            """
            Atom-resolved monopolar potental from instantaneous electric field.
            """

            vdp: Tensor
            """
            Atom-resolved dipolar potential from instantaneous electric field.
            """

            vqp: Tensor | None
            """
            Atom-resolved quadrupolar potential from electric field gradient.
            """

            def __init__(self, vat: Tensor, vdp: Tensor, vqp: Tensor | None) -> None:
                self.vat = vat
                self.vdp = vdp
                self.vqp = vqp

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            if self.__store is None:
                self.__store = self.Store(self.vat, self.vdp, self.vqp)

            slicer = slicers["atom"]
            self.vat = self.vat[[~conv, *slicer]]
            self.vdp = self.vdp[[~conv, *slicer, ...]]

            if self.vqp is not None:
                self.vqp = self.vqp[[~conv, *slicer, ...]]

        def restore(self) -> None:
            if self.__store is None:
                raise RuntimeError("Nothing to restore. Store is empty.")

            self.vat = self.__store.vat
            self.vdp = self.__store.vdp

    def __init__(
        self,
        field: Tensor,
        field_grad: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else field.device,
            dtype=dtype if dtype is None else field.dtype,
        )
        self.field = field
        self.field_grad = field_grad

    def get_cache(self, positions: Tensor, **_) -> Cache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        ElectricField.Cache
            Restart data for the interaction.

        Note
        ----
        If this interaction is evaluated within the `InteractionList`, `numbers`
        and `IndexHelper` will be passed as argument, too. The `**_` in the
        argument list will absorb those unnecessary arguments which are given
        as keyword-only arguments (see `Interaction.get_cache()`).
        """

        # (nbatch, natoms, 3) * (3) -> (nbatch, natoms)
        vat = torch.einsum("...ik,k->...i", positions, self.field)

        # (nbatch, natoms, 3)
        vdp = self.field.expand_as(positions)

        # (nbatch, natoms, 9)
        if self.field_grad is not None:
            if defaults.QP_SHAPE == 6:
                tmp = self.field_grad[torch.tril_indices(3, 3).unbind()]
            elif defaults.QP_SHAPE == 9:
                tmp = self.field_grad.flatten()
            else:
                raise ValueError
            vqp = tmp.expand((*positions.shape[:-1], tmp.shape[-1]))
        else:
            vqp = None

        return self.Cache(vat, vdp, vqp)

    def get_atom_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the monopolar contribution of the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """
        return -cache.vat * charges

    def get_dipole_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the dipolar contribution of the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic dipole moments of all atoms.
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """

        # equivalent: torch.sum(-cache.vdp * charges, dim=-1)
        return torch.einsum("...ix,...ix->...i", -cache.vdp, charges)

    def get_quadrupole_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the quadrupolar contribution of the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic dipole moments of all atoms.
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """
        if cache.vqp is None:
            return super().get_quadrupole_energy(charges, cache)

        # equivalent: torch.sum(-cache.vqp * charges, dim=-1)
        return torch.einsum("...ix,...ix->...i", -cache.vqp, charges)

    def get_atom_potential(self, _: Charges, cache: Cache) -> Tensor:
        """
        Calculate the electric field potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms (not required).
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field potential.
        """
        return -cache.vat

    def get_dipole_potential(self, _: Charges, cache: Cache) -> Tensor:
        """
        Calculate the electric field dipole potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms (not required).
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field dipole potential.
        """
        return -cache.vdp

    def get_quadrupole_potential(self, _: Charges, cache: Cache) -> Tensor | None:
        """
        Calculate the electric field quadrupole potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms (not required).
        cache : ElectricField.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field quadrupole potential.
        """
        return -cache.vqp if cache.vqp is not None else None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(field={self.field}, field_grad={self.field_grad})"

    def __repr__(self) -> str:
        return str(self)


def new_efield(
    field: Tensor,
    field_grad: Tensor | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ElectricField:
    """
    Create an instance of the electric field interaction.

    Parameters
    ----------
    field : Tensor
        Electric field vector consisting of the three cartesian components.
    device : torch.device | None, optional
        Device to store the tensor on. If `None` (default), the device is
        inferred from the `field` argument.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If `None` (default), the data type is inferred
        from the `field` argument.

    Returns
    -------
    ElectricField
        Instance of the electric field interaction.

    Raises
    ------
    RuntimeError
        Shape of `field` is not a vector of length 3.
    """
    if field.shape != torch.Size([3]):
        raise RuntimeError("Electric field must be a vector of length 3.")
    if field_grad is not None:
        if field_grad.shape != torch.Size((3, 3)):
            raise RuntimeError("Electric field gradient must be a 3 by 3 tensor.")

    if device is not None:
        if device != field.device:
            raise DeviceError(
                f"Passed device ({device}) and device of electric field "
                f"({field.device}) do not match."
            )
        if field_grad is not None:
            if device != field.device:
                raise DeviceError(
                    f"Passed device ({device}) and device of electric field "
                    f"gradient ({field_grad.device}) do not match."
                )

    if dtype is not None:
        if dtype != field.dtype:
            raise DtypeError(
                f"Passed dtype ({dtype}) and dtype of electric field "
                f"({field.dtype}) do not match."
            )
        if field_grad is not None:
            if dtype != field.dtype:
                raise DtypeError(
                    f"Passed dtype ({dtype}) and dtype of electric field "
                    f"gradient ({field_grad.dtype}) do not match."
                )

    return ElectricField(
        field,
        field_grad=field_grad,
        device=device if device is None else field.device,
        dtype=dtype if dtype is None else field.dtype,
    )
