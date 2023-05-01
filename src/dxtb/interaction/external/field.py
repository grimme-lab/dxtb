"""
External Fields
===============

Interaction of the charge density with external fields.
"""
from __future__ import annotations

import torch

from ..._types import Tensor, TensorLike
from ..base import Interaction

# from ..potential import Potential


class ElectricField(Interaction):
    """
    Instantaneous electric field.
    """

    field: Tensor
    """Instantaneous electric field vector."""

    __slots__ = ["field"]

    class Cache(Interaction.Cache, TensorLike):
        """
        Restart data for the electric field interaction.
        """

        vat: Tensor
        """
        Atom-resolved monopolar potental from instantaneous electric field.
        """

        vdp: Tensor
        """
        Atom-resolved dipolar potential from instantaneous electric field.
        """

        __slots__ = ["vat", "vdp"]

        def __init__(
            self,
            vat: Tensor,
            vdp: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else vat.device,
                dtype=dtype if dtype is None else vat.dtype,
            )
            self.vat = vat
            self.vdp = vdp

    def __init__(
        self,
        field: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            device=device if device is None else field.device,
            dtype=dtype if dtype is None else field.dtype,
        )
        self.field = field

    def get_cache(self, positions: Tensor, **_) -> Cache:
        """
        Create restart data for individual interactions.

        Returns
        -------
        Interaction.Cache
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

        return self.Cache(vat, vdp)

    def get_atom_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the electric field energy.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field interaction energies.
        """

        vat = -cache.vat * charges

        # TODO: Dipole integral required for dipolar potential
        # vdp = -cache.vdp * dp
        return vat

    def get_atom_potential(self, _: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the electric field potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atom-wise electric field potential.
        """
        return -cache.vat

        # return Potential(
        #     vat=-cache.vat,
        #     vdipole=-cache.vdp,
        #     vquadrupole=cache.vat.new_zeros([*cache.vat.shape, 6]),
        #     label=self.label,
        # )


def new_efield(
    field: Tensor,
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
        raise RuntimeError("Electric field must be a vector of lenght 3.")
    return ElectricField(
        field,
        device=device if device is None else field.device,
        dtype=dtype if dtype is None else field.dtype,
    )
