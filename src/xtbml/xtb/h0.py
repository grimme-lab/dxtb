from __future__ import annotations
import torch

from ..exlibs.tbmalt import batch
from ..basis.indexhelper import IndexHelper
from ..constants import EV2AU
from ..data import atomic_rad
from ..param import (
    get_pair_param,
    get_elem_param,
    get_elem_param_dict,
    get_elem_param_shells,
    Param,
)
from ..typing import Tensor

PAD = -999


def get_param_tensor_from_dict(
    numbers: Tensor, d: dict[int, list[bool | int | float]], dtype: torch.dtype
) -> Tensor:
    """Obtain parameters as tensor from dictionary. Taken from `IndexHelper`.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers.
    d : dict[int, list[bool|int|float]]
        Dictionary with the atomic numbers as keys and the values as lists.
    dtype : torch.dtype
        The `dtype` of the output tensor.

    Returns
    -------
    Tensor

    """
    return torch.tensor(
        [l for number in numbers for l in d.get(number.item(), [PAD])],
        dtype=dtype,
    )


class H0:

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    unique: Tensor
    """Unique species of the system."""

    positions: Tensor
    """Positions of the atoms in the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    selfenergy: Tensor
    """Self-energy of each species."""

    kcn: Tensor
    """Coordination number dependent shift of the self energy."""

    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""

    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""

    refocc: Tensor
    """Reference occupation numbers."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""

    valence: Tensor
    """Whether the shell belongs to the valence shell."""

    en: Tensor
    """Pauling electronegativity of each species."""

    rad: Tensor
    """Van-der-Waals radius of each species."""

    def __init__(self, numbers: Tensor, positions: Tensor, par: Param) -> None:
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.positions = positions
        self.par = par

        angular, valence = get_elem_param_shells(par.element, valence=True)
        self.ihelp = IndexHelper.from_numbers(numbers, angular)

        print("numbers", numbers)
        print("unique", self.unique)
        print("angular", self.ihelp.angular)
        print("unique_angular", self.ihelp.unique_angular)
        print("atom_to_unique", self.ihelp.atom_to_unique)
        print("shells_to_ushell", self.ihelp.shells_to_ushell)
        print("shell_index", self.ihelp.shell_index)
        print("shells_to_atom", self.ihelp.shells_to_atom)
        print("shells_per_atom", self.ihelp.shells_per_atom)
        print("orbitals_to_shell", self.ihelp.orbitals_to_shell)
        print("orbitals_per_shell", self.ihelp.orbitals_per_shell)
        print("\n")

        # atom-resolved parameters
        self.rad = atomic_rad[self.unique]
        self.en = get_elem_param(self.par.element, "en")[self.unique]

        # shell-resolved parameters
        self.kcn = self._get_elem_param("kcn")
        self.kpair = get_pair_param(self.par.hamiltonian.xtb.kpair)
        self.selfenergy = self._get_elem_param("levels")
        self.shpoly = self._get_elem_param("shpoly")
        self.refocc = self._get_elem_param("refocc")
        self.valence = get_param_tensor_from_dict(self.unique, valence, torch.bool)
        self.hscale = self._get_hscale()

        # unit conversion
        self.selfenergy = self.selfenergy * EV2AU
        self.kcn = self.kcn * EV2AU

    def _get_elem_param(self, key: str) -> Tensor:
        """Obtain element parameters for all species.

        Parameters
        ----------
        key : str
            Name of the parameter to be retrieved.

        Returns
        -------
        Tensor
            Parameters for each species.
        """
        return get_param_tensor_from_dict(
            self.unique,
            get_elem_param_dict(self.par.element, key),
            self.positions.dtype,
        )

    def _get_hscale(self) -> Tensor:
        """Obtain the off-site scaling factor for the Hamiltonian.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """

        _lsh2aqm = {
            0: "s",
            1: "p",
            2: "d",
            3: "f",
            4: "g",
        }

        def get_ksh(ushells: Tensor) -> Tensor:
            ksh = torch.ones((len(ushells), len(ushells)), dtype=self.positions.dtype)
            shell = self.par.hamiltonian.xtb.shell
            kpol = self.par.hamiltonian.xtb.kpol

            for i, ang_i in enumerate(ushells):
                ang_i = _lsh2aqm.get(int(ang_i.item()), PAD)

                if self.valence[i] == 0:
                    kii = kpol
                else:
                    kii = shell.get(f"{ang_i}{ang_i}", 1.0)

                for j, ang_j in enumerate(ushells):
                    ang_j = _lsh2aqm.get(int(ang_j.item()), PAD)

                    if self.valence[j] == 0:
                        kjj = kpol
                    else:
                        kjj = shell.get(f"{ang_j}{ang_j}", 1.0)

                    # only if both belong to the valence shell,
                    # we will read from the parametrization
                    if self.valence[i] == 1 and self.valence[j] == 1:
                        # check both "sp" and "ps"
                        ksh[i, j] = shell.get(
                            f"{ang_i}{ang_j}",
                            shell.get(
                                f"{ang_j}{ang_i}",
                                (kii + kjj) / 2.0,
                            ),
                        )
                    else:
                        ksh[i, j] = (kii + kjj) / 2.0

            return ksh

        ushells = self.ihelp.unique_angular
        if ushells.ndim > 1:
            ksh = torch.ones(
                (ushells.shape[0], ushells.shape[1], ushells.shape[1]),
                dtype=self.positions.dtype,
            )
            for _batch in range(ushells.shape[0]):
                ksh[_batch] = get_ksh(ushells[_batch])

            return ksh

        return get_ksh(ushells)

    # TODO: overlap
    def build(self, ovlp, cn: Tensor | None = None) -> Tensor:
        """Build the Hamiltonian."""
        real = self.ihelp.spread_atom_to_shell(self.numbers) != 0
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)

        zero = torch.tensor(
            0.0, device=self.positions.device, dtype=self.positions.dtype
        )

        # ----------------
        # Eq.29: H_(mu,mu)
        # ----------------
        selfenergy = self.ihelp.spread_ushell_to_shell(self.selfenergy)
        if cn is not None:
            cn = self.ihelp.spread_atom_to_shell(cn)
            kcn = self.ihelp.spread_ushell_to_shell(self.kcn)

            # formula differs from paper to be consistent with GFN2 -> "kcn" adapted
            selfenergy -= kcn * cn

        # ----------------------
        # Eq.24: PI(R_AB, l, l')
        # ----------------------
        idx = self.ihelp.shells_to_atom
        inp = self.positions
        size = [*idx.size(), inp.size(-1)]

        idx = idx.unsqueeze(-1).expand(*size)
        positions = torch.where(
            idx >= 0,
            torch.gather(inp, -2, torch.where(idx >= 0, idx, 0)),
            zero,
        )

        # TODO: spread only at distances
        distances = torch.cdist(positions, positions, p=2)
        rad = self.ihelp.spread_uspecies_to_shell(self.rad)
        rr = torch.where(
            mask,
            torch.sqrt(distances / (rad.unsqueeze(-1) + rad.unsqueeze(-2))),
            zero,
        )

        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)

        var_pi = (1.0 + shpoly.unsqueeze(-1) * rr) * (1.0 + shpoly.unsqueeze(-2) * rr)

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        var_x = torch.where(
            mask,
            1.0
            + self.par.hamiltonian.xtb.enscale
            * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
            zero,
        )

        # --------------------
        # Eq.23: K_{AB}^{l,l'}
        # --------------------
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        kpair = self.ihelp.spread_ushell_to_shell(self.kpair, dim=(-2, -1))
        valence = self.ihelp.spread_ushell_to_shell(self.valence)
        # torch.index_select(x[index], -1, index)

        var_k = torch.where(
            valence.unsqueeze(-1) * valence.unsqueeze(-2),
            hscale * kpair * var_x,
            hscale,
        )

        # ------------
        # Eq.23: H_EHT
        # ------------
        selfenergy = torch.where(
            mask, 0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)), zero
        )

        # TODO
        var_s = ovlp

        mask.diagonal(dim1=-2, dim2=-1).fill_(False)
        h0 = torch.where(mask, var_pi * var_k * selfenergy, selfenergy)

        print("h0", h0)
        h = self.ihelp.spread_shell_to_orbital(h0, dim=(-2, -1))
        return var_s * h
