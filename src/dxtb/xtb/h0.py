"""
The GFN1-xTB Hamiltonian.
"""
from __future__ import annotations

import torch

from .._types import Tensor, TensorLike
from ..basis import IndexHelper
from ..constants import EV2AU
from ..data import atomic_rad
from ..param import Param, get_elem_param, get_elem_valence, get_pair_param
from ..utils import real_pairs, symmetrize

PAD = -1
"""Value used for padding of tensors."""


class Hamiltonian(TensorLike):
    """Hamiltonian from parametrization."""

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""
    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    hscale: Tensor
    """Off-site scaling factor for the Hamiltonian."""
    kcn: Tensor
    """Coordination number dependent shift of the self energy."""
    kpair: Tensor
    """Element-pair-specific parameters for scaling the Hamiltonian."""
    refocc: Tensor
    """Reference occupation numbers."""
    selfenergy: Tensor
    """Self-energy of each species."""
    shpoly: Tensor
    """Polynomial parameters for the distant dependent scaling."""
    valence: Tensor
    """Whether the shell belongs to the valence shell."""

    en: Tensor
    """Pauling electronegativity of each species."""
    rad: Tensor
    """Van-der-Waals radius of each species."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp

        if self.par.hamiltonian is None:
            raise RuntimeError("Parametrization does not specify a Hamiltonian.")

        # atom-resolved parameters
        self.rad = atomic_rad[self.unique].type(self.dtype).to(device=self.device)
        self.en = self._get_elem_param("en")

        # shell-resolved element parameters
        self.kcn = self._get_elem_param("kcn")
        self.selfenergy = self._get_elem_param("levels")
        self.shpoly = self._get_elem_param("shpoly")
        self.refocc = self._get_elem_param("refocc")
        self.valence = self._get_elem_valence()

        # shell-pair-resolved pair parameters
        self.hscale = self._get_hscale()
        self.kpair = self._get_pair_param(self.par.hamiltonian.xtb.kpair)

        # unit conversion
        self.selfenergy = self.selfenergy * EV2AU
        self.kcn = self.kcn * EV2AU

        if any(
            tensor.dtype != self.dtype
            for tensor in (
                self.hscale,
                self.kcn,
                self.kpair,
                self.refocc,
                self.selfenergy,
                self.shpoly,
                self.en,
                self.rad,
            )
        ):
            raise ValueError("All tensors must have same dtype")

        if any(
            tensor.device != self.device
            for tensor in (
                self.numbers,
                self.unique,
                self.ihelp,
                self.hscale,
                self.kcn,
                self.kpair,
                self.refocc,
                self.selfenergy,
                self.shpoly,
                self.valence,
                self.en,
                self.rad,
            )
        ):
            raise ValueError("All tensors must be on the same device")

    def get_occupation(self) -> Tensor:
        """
        Obtain the reference occupation numbers for each orbital.
        """

        refocc = self.ihelp.spread_ushell_to_orbital(self.refocc)
        orb_per_shell = self.ihelp.spread_shell_to_orbital(
            self.ihelp.orbitals_per_shell
        )

        return torch.where(
            orb_per_shell != 0, refocc / orb_per_shell, refocc.new_tensor(0)
        )

    def _get_elem_param(self, key: str) -> Tensor:
        """Obtain element parameters for species.

        Parameters
        ----------
        key : str
            Name of the parameter to be retrieved.

        Returns
        -------
        Tensor
            Parameters for each species.
        """

        return get_elem_param(
            self.unique,
            self.par.element,
            key,
            pad_val=PAD,
            device=self.device,
            dtype=self.dtype,
        )

    def _get_elem_valence(self) -> Tensor:
        """Obtain "valence" parameters for shells of species.

        Returns
        -------
        Tensor
            Valence parameters for each species.
        """

        return get_elem_valence(
            self.unique,
            self.par.element,
            pad_val=PAD,
            device=self.device,
            dtype=torch.bool,
        )

    def _get_pair_param(self, pair: dict[str, float]) -> Tensor:
        """Obtain element-pair-specific parameters for all species.

        Parameters
        ----------
        pair : dict[str, float]
            Pair parametrization.

        Returns
        -------
        Tensor
            Pair parameters for each species.
        """

        return get_pair_param(
            self.unique.tolist(), pair, device=self.device, dtype=self.dtype
        )

    def _get_hscale(self) -> Tensor:
        """Obtain the off-site scaling factor for the Hamiltonian.

        Returns
        -------
        Tensor
            Off-site scaling factor for the Hamiltonian.
        """
        angular2label = {
            0: "s",
            1: "p",
            2: "d",
            3: "f",
            4: "g",
        }

        def get_ksh(ushells: Tensor) -> Tensor:
            if self.par.hamiltonian is None:
                raise RuntimeError("No Hamiltonian specified.")

            ksh = torch.ones(
                (len(ushells), len(ushells)), dtype=self.dtype, device=self.device
            )
            shell = self.par.hamiltonian.xtb.shell
            kpol = self.par.hamiltonian.xtb.kpol

            for i, ang_i in enumerate(ushells):
                ang_i = angular2label.get(int(ang_i.item()), PAD)

                if self.valence[i] == 0:
                    kii = kpol
                else:
                    kii = shell.get(f"{ang_i}{ang_i}", 1.0)

                for j, ang_j in enumerate(ushells):
                    ang_j = angular2label.get(int(ang_j.item()), PAD)

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
                dtype=self.dtype,
                device=self.device,
            )
            for _batch in range(ushells.shape[0]):
                ksh[_batch] = get_ksh(ushells[_batch])

            return ksh

        return get_ksh(ushells)

    def build(
        self, positions: Tensor, overlap: Tensor, cn: Tensor | None = None
    ) -> Tensor:
        """
        Build the xTB Hamiltonian.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of molecular structure.
        overlap : Tensor
            Overlap matrix.
        cn : Tensor | None, optional
            Coordination number. Defaults to `None`.

        Returns
        -------
        Tensor
            Hamiltonian (always symmetric).
        """
        if self.par.hamiltonian is None:
            raise RuntimeError("No Hamiltonian specified.")

        # masks
        mask_atom_diagonal = real_pairs(self.numbers, diagonal=True)
        mask_shell = real_pairs(self.ihelp.spread_atom_to_shell(self.numbers))
        mask_shell_diagonal = self.ihelp.spread_atom_to_shell(
            mask_atom_diagonal, dim=(-2, -1)
        )

        zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # ----------------
        # Eq.29: H_(mu,mu)
        # ----------------
        if cn is None:
            cn = torch.zeros_like(self.numbers).type(self.dtype)

        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)

        # formula differs from paper to be consistent with GFN2 -> "kcn" adapted
        selfenergy = self.ihelp.spread_ushell_to_shell(
            self.selfenergy
        ) - kcn * self.ihelp.spread_atom_to_shell(cn)

        # ----------------------
        # Eq.24: PI(R_AB, l, l')
        # ----------------------
        distances = torch.cdist(
            positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
        )
        rad = self.ihelp.spread_uspecies_to_atom(self.rad)
        rr = torch.where(
            mask_atom_diagonal,
            distances / (rad.unsqueeze(-1) + rad.unsqueeze(-2)),
            distances.new_tensor(torch.finfo(distances.dtype).eps),
        )
        rr_shell = self.ihelp.spread_atom_to_shell(
            torch.sqrt(rr),
            (-2, -1),
        )

        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)
        var_pi = (1.0 + shpoly.unsqueeze(-1) * rr_shell) * (
            1.0 + shpoly.unsqueeze(-2) * rr_shell
        )

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        var_x = torch.where(
            mask_shell_diagonal,
            1.0
            + self.par.hamiltonian.xtb.enscale
            * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
            zero,
        )

        # --------------------
        # Eq.23: K_{AB}^{l,l'}
        # --------------------
        kpair = self.ihelp.spread_uspecies_to_shell(self.kpair, dim=(-2, -1))
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        valence = self.ihelp.spread_ushell_to_shell(self.valence)

        var_k = torch.where(
            valence.unsqueeze(-1) * valence.unsqueeze(-2),
            hscale * kpair * var_x,
            hscale,
        )

        # ------------
        # Eq.23: H_EHT
        # ------------
        var_h = torch.where(
            mask_shell,
            0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)),
            zero,
        )

        hcore = self.ihelp.spread_shell_to_orbital(
            torch.where(
                mask_shell_diagonal,
                var_pi * var_k * var_h,  # scale only off-diagonals
                var_h,
            ),
            dim=(-2, -1),
        )
        h = hcore * overlap

        # force symmetry to avoid problems through numerical errors
        return symmetrize(h)

    def get_gradient(
        self,
        positions: Tensor,
        overlap: Tensor,
        doverlap: Tensor,
        pmat: Tensor,
        wmat: Tensor,
        pot: Tensor,
        cn: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate gradient of the full Hamiltonian with respect ot atomic positions.

        Parameters
        ----------
        positions : Tensor
            Atomic positions of molecular structure.
        overlap : Tensor
            Overlap matrix.
        doverlap : Tensor
            Derivative of the overlap matrix.
        pmat : Tensor
            Density matrix.
        wmat : Tensor
            Energy-weighted density.
        pot : Tensor
            Self-consistent electrostatic potential.
        cn : Tensor
            Coordination number.

        Returns
        -------
        tuple[Tensor, Tensor]
            Derivative of energy with respect to coordination number (first
            tensor) and atomic positions (second tensor).
        """
        if self.par.hamiltonian is None:
            raise RuntimeError("No Hamiltonian specified.")

        # masks
        mask_atom = real_pairs(self.numbers)
        mask_atom_diagonal = real_pairs(self.numbers, diagonal=True)

        mask_shell = real_pairs(self.ihelp.spread_atom_to_shell(self.numbers))
        mask_shell_diagonal = self.ihelp.spread_atom_to_shell(
            mask_atom_diagonal, dim=(-2, -1)
        )

        mask_orb_diagonal = self.ihelp.spread_atom_to_orbital(
            mask_atom_diagonal, dim=(-2, -1)
        )

        zero = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # --------------------
        # Eq.28: X(EN_A, EN_B)
        # --------------------
        en = self.ihelp.spread_uspecies_to_shell(self.en)
        var_x = torch.where(
            mask_shell_diagonal,
            1.0
            + self.par.hamiltonian.xtb.enscale
            * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
            zero,
        )

        # --------------------
        # Eq.23: K_{AB}^{l,l'}
        # --------------------
        kpair = self.ihelp.spread_uspecies_to_shell(self.kpair, dim=(-2, -1))
        hscale = self.ihelp.spread_ushell_to_shell(self.hscale, dim=(-2, -1))
        valence = self.ihelp.spread_ushell_to_shell(self.valence)

        var_k = torch.where(
            valence.unsqueeze(-1) * valence.unsqueeze(-2),
            hscale * kpair * var_x,
            hscale,
        )

        # ----------------------
        # Eq.24: PI(R_AB, l, l')
        # ----------------------
        distances = torch.cdist(
            positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
        )
        rad = self.ihelp.spread_uspecies_to_atom(self.rad)
        rr = torch.where(
            mask_atom_diagonal,
            distances / (rad.unsqueeze(-1) + rad.unsqueeze(-2)),
            distances.new_tensor(torch.finfo(distances.dtype).eps),
        )
        rr_shell = self.ihelp.spread_atom_to_shell(
            torch.sqrt(rr),
            (-2, -1),
        )

        shpoly = self.ihelp.spread_ushell_to_shell(self.shpoly)
        shpoly_a = shpoly.unsqueeze(-1)
        tmp_a = 1.0 + shpoly_a * rr_shell
        shpoly_b = shpoly.unsqueeze(-2)
        tmp_b = 1.0 + shpoly_b * rr_shell
        var_pi = tmp_a * tmp_b

        # ------------
        # Eq.23: H_EHT
        # ------------

        # `kcn` differs from paper (Eq.29) to be consistent with GFN2
        kcn = self.ihelp.spread_ushell_to_shell(self.kcn)
        selfenergy = self.ihelp.spread_ushell_to_shell(
            self.selfenergy
        ) - kcn * self.ihelp.spread_atom_to_shell(cn)

        var_h = torch.where(
            mask_shell,
            0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)),
            zero,
        )

        hcore = self.ihelp.spread_shell_to_orbital(
            torch.where(
                mask_shell_diagonal,
                var_pi * var_k * var_h,  # scale only off-diagonals
                var_h,
            ),
            dim=(-2, -1),
        )

        # ----------------------------------------------------------------------
        # Derivative of the electronic energy w.r.t. the atomic positions r
        # ----------------------------------------------------------------------
        # dE/dr = dE_EHT/dr * dE_coulomb/dr * dL_constraint/dr
        #       = [2*P*H - 2*W - P*(V + V^T)] * dS/dr + 2*P*H*S * dPI/dr / PI
        # ----------------------------------------------------------------------

        # ------------------------------------------------------------
        # derivative of Eq.24: PI(R_AB, l, l') -> dPI/dr (without rij)
        # ------------------------------------------------------------
        distances_shell = self.ihelp.spread_atom_to_shell(distances, (-2, -1))
        dvar_pi = torch.where(
            mask_shell_diagonal,
            (tmp_a * shpoly_b + tmp_b * shpoly_a)
            * rr_shell
            * 0.5
            / torch.pow(distances_shell, 2.0),
            zero,
        )

        # xTB Hamiltonian (without overlap, Hcore) times density matrix
        ph = pmat * hcore

        # E_EHT derivative for scaling function `PI` (2*P*H*S * dPI/dr / PI)
        dpi = (
            2
            * self.ihelp.reduce_orbital_to_shell(ph * overlap, dim=(-2, -1))
            * dvar_pi
            / var_pi
        )

        # factors for all derivatives of the overlap (2*P*H - 2*W - P*(V + V^T))
        sval = torch.where(
            mask_orb_diagonal,
            2 * (ph - wmat) - pmat * (pot.unsqueeze(-1) + pot.unsqueeze(-2)),
            zero,
        )

        # distance vector from dR_AB/dr_a [n_batch, atoms_i, atoms_j, 3]
        rij = torch.where(
            mask_atom.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            positions.new_tensor(0.0),
        )

        # reduce to atoms
        dpi = self.ihelp.reduce_shell_to_atom(dpi, dim=(-2, -1))

        # contract within orbital representation
        ss_orb = sval.unsqueeze(-1) * doverlap

        # instead of looping over atom and orbital combinations to obtain
        # correct overlap chunks, mask out irrelevant ones
        natm = positions.shape[-2]
        batch_mode = len(positions.shape) > 2

        # vectorised masking in chunks over atom-pairs
        if batch_mode:
            gradient = torch.zeros([ss_orb.shape[0], natm, 3])
        else:
            gradient = torch.zeros([natm, 3])

        o2a = self.ihelp.orbitals_per_atom
        master = ss_orb.float()
        _zero = gradient.new_tensor(0.0)

        for i in range(natm):
            for j in range(natm):
                mask_ij = (o2a.unsqueeze(-2) == i) & (o2a.unsqueeze(-1) == j)

                if batch_mode:
                    mask_ij = mask_ij.unsqueeze(-1).repeat(1, 1, 1, 3)
                    master_ij = torch.where(mask_ij, master, _zero)
                    gradient[..., i, :] += torch.einsum("bijm->bm", master_ij)
                    # [bs, norb, norb, 3] -> [bs, 3]
                else:
                    # spread to all atom-pairs
                    mask_ij = mask_ij.unsqueeze(-1).repeat(1, 1, 3)
                    # mask all non-contributing pairs
                    master_ij = torch.where(mask_ij, master, _zero)
                    # sum over orbital- and atom-wise contributions
                    gradient[..., i, :] += torch.einsum("ijm->m", master_ij)
            # NOTE: pytorch autograd requires a lot of RAM here (for large molecules)

        # add E_EHT contribution
        gradient += torch.sum(dpi.unsqueeze(-1) * rij, dim=-2)

        # ----------------------------------------------------------------------
        # Derivative of the electronic energy w.r.t. the coordination number
        # ----------------------------------------------------------------------
        # E = P * H = P * 0.5 * (H_mm(CN) + H_nn(CN)) * S * F
        # -> with: H_mm(CN) = se_mm(CN) = selfenergy - kcn * CN
        #          F = PI(R_AB, l, l') * K_{AB}^{l,l'} * X(EN_A, EN_B)
        # ----------------------------------------------------------------------

        # `kcn` differs from paper (Eq.29) to be consistent with GFN2
        dsedcn = -self.ihelp.spread_ushell_to_shell(self.kcn).unsqueeze(-2)

        # avoid symmetric matrix by only passing `dsedcn` vector, which must be
        # unsqueeze(-2)'d for batched calculations
        dhdcn = torch.where(
            mask_shell_diagonal,
            dsedcn * var_pi * var_k,  # only scale off-diagonals
            dsedcn,
        )

        # reduce orbital-resolved `P*S` for mult with shell-resolved `dhdcn`
        dcn = self.ihelp.reduce_orbital_to_shell(pmat * overlap, dim=(-2, -1)) * dhdcn

        # reduce to atoms and sum for vector (requires non-symmetric matrix)
        dedcn = self.ihelp.reduce_shell_to_atom(dcn, dim=(-2, -1))

        return dedcn.sum(-2), gradient

    def to(self, device: torch.device) -> Hamiltonian:
        """
        Returns a copy of the `Hamiltonian` instance on the specified device.

        This method creates and returns a new copy of the `Hamiltonian` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        Hamiltonian
            A copy of the `Hamiltonian` instance placed on the specified device.

        Notes
        -----
        If the `Hamiltonian` instance is already on the desired device `self`
        will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.numbers.to(device=device),
            self.par,
            self.ihelp.to(device=device),
            device=device,
        )

    def type(self, dtype: torch.dtype) -> Hamiltonian:
        """
        Returns a copy of the `Hamiltonian` instance with specified floating point type.
        This method creates and returns a new copy of the `Hamiltonian` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Type of the floating point numbers used by the `Hamiltonian` instance.

        Returns
        -------
        Hamiltonian
            A copy of the `Hamiltonian` instance with the specified dtype.

        Notes
        -----
        If the `Hamiltonian` instance has already the desired dtype `self` will
        be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.numbers.type(dtype=torch.long),
            self.par,
            self.ihelp.type(dtype=dtype),
            dtype=dtype,
        )
