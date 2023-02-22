"""
On-site third-order electrostatic energy (ES3)
==============================================

This module implements the third-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.thirdorder as es3
>>> from xtbml.param import GFN1_XTB, get_element_param, get_elem_angular
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> qat = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> hubbard_derivs = get_element_param(GFN1_XTB.element, "gam3")
>>> ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(GFN1_XTB.element))
>>> es = es3.ES3(positions, hubbard_derivs)
>>> cache = es.get_cache(ihelp)
>>> e = es.get_atom_energy(qat, cache)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0155669)
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike
from ..basis import IndexHelper
from ..interaction import Interaction
from ..param import Param, get_elem_param

__all__ = ["ES3", "new_es3"]


class ES3(Interaction):
    """
    On-site third-order electrostatic energy.
    """

    hubbard_derivs: Tensor
    """Hubbard derivatives of all atoms."""

    __slots__ = ["hubbard_derivs"]

    class Cache(Interaction.Cache, TensorLike):
        """
        Restart data for the ES3 interaction.
        """

        hd: Tensor
        """Spread Hubbard derivatives of all atoms (not only unique)."""

        __slots__ = ["hd"]

        def __init__(
            self,
            hd: Tensor,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            super().__init__(
                device=device if device is None else hd.device,
                dtype=dtype if dtype is None else hd.dtype,
            )
            self.hd = hd

    def __init__(
        self,
        hubbard_derivs: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.hubbard_derivs = hubbard_derivs

    def get_cache(self, ihelp: IndexHelper, **_) -> Cache:
        """
        Create restart data for individual interactions.

        Parameters
        ----------
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Interaction.Cache
            Restart data for the interaction.

        Note
        ----
        If this `ES3` interaction is evaluated within the `InteractionList`,
        `numbers` and `positions` will be passed as argument, too. The `**_` in
        the argument list will absorb those unnecessary arguments which are
        given as keyword-only arguments (see `Interaction.get_cache()`).
        """

        return self.Cache(ihelp.spread_uspecies_to_atom(self.hubbard_derivs))

    def get_atom_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        """
        Calculate the third-order electrostatic energy.

        Implements Eq.30 of the following paper:
        - C. Bannwarth, E. Caldeweyher, S. Ehlert, A. Hansen, P. Pracht,
        J. Seibert, S. Spicher and S. Grimme, *WIREs Computational Molecular
        Science*, **2020**, 11, e1493. DOI: `10.1002/wcms.1493
        <https://wires.onlinelibrary.wiley.com/doi/10.1002/wcms.1493>`__

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction energies.
        """

        return cache.hd * torch.pow(charges, 3.0) / 3.0

    def get_atom_potential(self, charges: Tensor, cache: Cache) -> Tensor:
        """Calculate the third-order electrostatic potential.

        Parameters
        ----------
        charges : Tensor
            Atomic charges of all atoms.
        cache : Interaction.Cache
            Restart data for the interaction.

        Returns
        -------
        Tensor
            Atomwise third-order Coulomb interaction potential.
        """

        return cache.hd * torch.pow(charges, 2.0)


def new_es3(
    numbers: Tensor,
    par: Param,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES3 | None:
    """
    Create new instance of ES3.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    ES3 | None
        Instance of the ES3 class or `None` if no ES3 is used.
    """

    if hasattr(par, "thirdorder") is False or par.thirdorder is None:
        return None

    if par.thirdorder.shell is True:
        raise NotImplementedError(
            "Shell-resolved third order electrostatics are not implemented. "
            "Set `thirdorder.shell` parameter to `False`."
        )

    hubbard_derivs = get_elem_param(
        torch.unique(numbers),
        par.element,
        "gam3",
    )

    return ES3(hubbard_derivs, device=device, dtype=dtype)
