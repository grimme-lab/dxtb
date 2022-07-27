""" Definition of basic data classes """
from __future__ import annotations
from math import log10, sqrt
import torch
from typing import Union, Dict, List
from xtbml.basis.indexhelper import IndexHelper

from xtbml.param.util import get_elem_param, get_elem_pqn, get_elem_valence

from ..basis import slater, orthogonalize
from ..param import Param, Element
from ..constants import UINT8 as DTYPE_INT
from ..constants import PSE
from ..typing import Tensor

MAXG = 12
"""Maximum contraction length of basis functions. The limit is chosen as twice the maximum size returned by the STO-NG expansion"""

_aqm2lsh = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}


class Bas:
    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        acc: float = 1.0,
    ) -> None:
        self.intcut = integral_cutoff(acc)

        self.ngauss = get_elem_param(numbers, par.element, "ngauss")
        self.slater = get_elem_param(
            numbers, par.element, "slater", dtype=dtype, device=device
        )
        self.valence = get_elem_valence(numbers, par.element, dtype=torch.uint8)
        self.pqn = get_elem_pqn(numbers, par.element)

    def create_cgtos(self, ihelp: IndexHelper):
        cgto = []
        coeffs = []
        alphas = []

        # maybe this can be batched too, but this loop is rather small
        # so it is probably not worth it
        for i in range(len(self.slater)):
            ret = slater.to_gauss(
                self.ngauss[i], self.pqn[i], ihelp.unique_angular[i], self.slater[i]
            )
            cgtoi = Cgto_Type(ihelp.unique_angular[i], *ret)

            # FIXME: this only works for GFN1 with H being the only element
            # with a non-valence shell
            if self.valence[i].item() == 0:
                cgtoi = Cgto_Type(
                    ihelp.unique_angular[i],
                    *orthogonalize(
                        ihelp.unique_angular[i],
                        (cgto[i - 1].alpha, cgtoi.alpha),
                        (cgto[i - 1].coeff, cgtoi.coeff),
                    ),
                )

            cgto.append(cgtoi)
            alphas.append(cgtoi.alpha)
            coeffs.append(cgtoi.coeff)

        return cgto, alphas, coeffs

    def create_umap(self, ihelp: IndexHelper):
        torch.set_printoptions(linewidth=200)

        # offsets to avoid duplication on addition
        offset1 = 100
        offset2 = 1000

        # orbital combinations (duplicates between atoms avoided by offset)
        sh2orb = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)
        sh2ush = ihelp.spread_shell_to_orbital(ihelp.shells_to_ushell)
        orbs = sh2ush + sh2orb * offset1

        # extra offset along only one dimension to distinguish (n, m) and
        # (m, n) of the same orbital block (e.g. 1x3 sp and 3x1 ps block)
        u = orbs.unsqueeze(-2) + orbs.unsqueeze(-1) * offset2

        # NOTE: atom distinction should already be encoded through `offset1`
        # unique atom combinations: offset to remove duplicates across atoms
        # atoms = ihelp.spread_atom_to_orbital(ihelp.atom_to_unique)
        # atoms = atoms.unsqueeze(-2) * offset2 + atoms.unsqueeze(-1) * offset3
        # u = atoms + orbs

        # get unique map
        _, umap = torch.unique(u, return_inverse=True)

        # number of unqiue shells
        n_uangular = len(ihelp.unique_angular)

        # calculated number of unique combinations of unique shells
        n_unique_pairs = torch.max(umap) + 1

        # check against theoretical number
        n_uangular_comb = torch.sum(torch.arange(1, n_uangular + 1))
        if n_unique_pairs < n_uangular_comb:
            raise ValueError(
                f"Internal error: {n_uangular_comb- n_unique_pairs} missing unique pairs."
            )

        return umap, n_unique_pairs


class Cgto_Type:
    """Contracted Gaussian type basis function"""

    def __init__(self, ang, alpha, coeff):
        # Angular momentum of this basis function
        self.ang = ang
        # Contraction length of this basis function
        self.nprim = min(alpha.shape[-1], coeff.shape[-1])
        # Exponent of the primitive Gaussian functions
        self.alpha = alpha
        # Contraction coefficients of the primitive Gaussian functions,
        # might contain normalization
        self.coeff = coeff

    def __str__(self):
        return f"cgto( l:{self.ang} | ng:{self.nprim} | alpha:{self.alpha} | coeff:{self.coeff} )"

    def __repr__(self):
        return self.__str__()


class Basis:
    """
    Atomic orbital basis set definition.
    """

    cgto: Dict[str, List[Cgto_Type]]
    """Definition of the atom centered basis functions for each species"""

    shells: Dict[str, int]
    """Number of shells for each species"""

    nsh_per_atom: List[int]
    """Number of shells per atom in basis set."""

    nsh_per_id: Dict[str, int]
    """Number of shells per unique atom (=id) in basis set."""

    ish_at: torch.Tensor
    """Index offset for each atom in the shell space"""

    nsh_tot: int = 0
    """Total number of shells in basis set."""

    nao_tot: int = 0
    """Total number of spherical atomic orbitals in basis set."""

    maxl: int = 0
    """Maximum angular momentum of all basis functions, used to determine scratch size in integral calculation"""

    intcut: float = 0.0
    """Integral cutoff as maximum exponent of Gaussian product theoreom to consider"""

    min_alpha: float = float("inf")
    """Smallest primitive exponent in the basis set"""

    # xtbML               tblite
    # ---------------------------
    # shells              nshells
    # shells              nshell
    # shells              nsh_id
    # nsh_per_atom        nsh_at
    # nsh_tot             nsh
    # nao_tot             nao
    # ish_at              ish_at

    def __init__(self, numbers: Tensor, par: Param, acc: float = 1.0) -> None:
        self.nsh_per_atom = []
        self.nsh_per_id = {}
        self.shells = {}
        self.cgto = {}

        self.intcut = integral_cutoff(acc)

        self.symbols = [PSE.get(x, "X") for x in numbers.tolist()]
        self.usymbols = [
            PSE.get(x, "X") for x in torch.unique(numbers[numbers.ne(0)]).tolist()
        ]

        for isp in self.symbols:
            record = par.element[isp]
            self.cgto[isp] = _process_record(record)
            self.shells[isp] = len(record.shells)
            self.nsh_per_atom.append(len(record.shells))

        self.nsh_tot = sum(self.nsh_per_atom)

        # Create mapping between atoms and shells
        # (offset array for indexing (e.g. selfenergy) later)
        self.ish_at = torch.zeros(torch.numel(numbers), dtype=DTYPE_INT)
        counter = 0
        for i in range(torch.numel(numbers)):
            self.ish_at[i] = counter
            counter += self.nsh_per_atom[i]

        # Make count of spherical orbitals for each shell
        self.nao_sh = torch.zeros(self.nsh_tot, dtype=DTYPE_INT)
        counter = 0
        for i, element in enumerate(self.symbols):
            counter = int(self.ish_at[i].item())
            for ish in range(self.nsh_per_atom[i]):
                self.nao_sh[counter + ish] = 2 * self.cgto[element][ish].ang + 1

        self.nao_tot = int(torch.sum(self.nao_sh).item())

        # Create mapping between shells and spherical orbitals
        self.iao_sh = torch.zeros(self.nsh_tot, dtype=DTYPE_INT)
        counter = 0
        for i in range(self.nsh_tot):
            self.iao_sh[i] = counter
            counter += self.nao_sh[i]

        counter = 0
        for i, element in enumerate(self.symbols):
            for ish in range(self.shells[element]):
                iat = int(self.ish_at[i].item())
                self.iao_sh[ish + iat] = counter
                counter += 2 * self.cgto[element][ish].ang + 1

        # Generate min_alpha and maxl
        for i, element in enumerate(self.usymbols):
            for ish in range(self.shells[element]):
                cgto = self.cgto[element][ish]
                self.maxl = max(self.maxl, cgto.ang)
                self.min_alpha = min(
                    self.min_alpha, torch.min(cgto.alpha[: cgto.nprim]).item()
                )


def _process_record(record: Element) -> List[Cgto_Type]:
    """
    Create basis set from element record
    """

    cgto = []

    nsh = len(record.shells)
    ang_idx = nsh * [-1]
    ortho = nsh * [None]

    lsh = [_aqm2lsh[shell[-1]] for shell in record.shells]
    pqn = [int(shell[0]) for shell in record.shells]

    for ish in range(nsh):
        il = lsh[ish]
        ng = record.ngauss[ish]

        # print("ish:", ish, "il:", il)
        # print("ang_idx:", ang_idx[il])

        if ang_idx[il] >= 0:
            ortho[ish] = ang_idx[il]
        else:
            ang_idx[il] = ish

        # print("ng:", ng, "pqn:", pqn[ish], "ang:", il, "slater: ", record.slater[ish])
        # print(slater.to_gauss(ng, pqn[ish], il, torch.tensor(record.slater[ish])))
        cgtoi = Cgto_Type(
            il, *slater.to_gauss(ng, pqn[ish], il, torch.tensor(record.slater[ish]))
        )
        cgto.append(cgtoi)

    for ish in range(nsh):
        if ortho[ish] is not None:
            cgto[ish] = Cgto_Type(
                lsh[ish],
                *orthogonalize(
                    lsh[ish],
                    (cgto[ortho[ish]].alpha, cgto[ish].alpha),
                    (cgto[ortho[ish]].coeff, cgto[ish].coeff),
                ),
            )

    return cgto


def get_cutoff(basis: Basis, acc: Union[None, float] = None) -> float:
    max_cutoff = 40.0

    if acc is not None:
        intcut = integral_cutoff(acc)
    else:
        intcut = basis.intcut

    return min(sqrt(2 * intcut / basis.min_alpha), max_cutoff)


def integral_cutoff(acc: float) -> float:
    """Create integral cutoff from accuracy value
    Args:
        acc (float, optional): Accuracy for the integral cutoff. Defaults to None.
    Returns:
        float: Integral cutoff
    """
    min_intcut = 5.0
    max_intcut = 25.0
    max_acc = 1.0e-4
    min_acc = 1.0e3
    intcut = clip(
        max_intcut - 10 * log10(clip(acc, min_acc, max_acc)), min_intcut, max_intcut
    )
    return intcut


def clip(val: float, min_val: float, max_val: float):
    return min(max(val, min_val), max_val)
