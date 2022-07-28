""" Definition of basic data classes """
from __future__ import annotations
from math import log10, sqrt
import torch
from typing import Union, Dict, List
from xtbml.basis.indexhelper import IndexHelper


from ..basis import slater, orthogonalize
from ..param import Param, Element, get_elem_param, get_elem_pqn, get_elem_valence
from ..constants import UINT8 as DTYPE_INT
from ..constants import PSE
from ..typing import Tensor
from ..utils import timing

MAXG = 12
"""Maximum contraction length of basis functions. The limit is chosen as twice the maximum size returned by the STO-NG expansion"""

_aqm2lsh = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}

# fmt: off
primes = torch.tensor(
    [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987]
)
"""The first 300 prime numbers"""
# fmt: on


class Bas:
    """Atomic orbital basis set."""

    angular: Tensor
    """Angular momenta of all shells (from `IndexHelper`)"""

    ngauss: Tensor
    """Number of Gaussians used in expansion from Slater orbital."""

    slater: Tensor
    """Exponent of Slater function."""

    pqn: Tensor
    """Principal quantum number of each shell"""

    valence: Tensor
    """Whether the shell is part of the valence shell."""

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        angular: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        self.angular = angular
        self.ngauss = get_elem_param(numbers, par.element, "ngauss")
        self.slater = get_elem_param(
            numbers, par.element, "slater", dtype=dtype, device=device
        )
        self.pqn = get_elem_pqn(numbers, par.element)
        self.valence = get_elem_valence(numbers, par.element)

    @timing
    def create_cgtos(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Create contracted Gaussian type orbitals from parametrization.

        Returns
        -------
        (list[Tensor], list[Tensor])
            List of primitive Gaussian exponents and contraction coefficients
            for the orthonormalized basis functions for each shell.
        """
        coeffs = []
        alphas = []

        # maybe this can be batched too, but this loop is rather small
        # so it is probably not worth it
        for i in range(self.angular.size(0)):
            alpha, coeff = slater.to_gauss(
                self.ngauss[i], self.pqn[i], self.angular[i], self.slater[i]
            )

            # FIXME:
            # This only works for GFN1 with H being the only element
            # with a non-valence shell. Otherwise the generation of
            # self.valence is not correct.
            # The correct way would be a map to show which orbital needs
            # to be orthogonalized w.r.t. another one.
            # Example: Si with 3s, 3p, 3d, 4p
            # angular = [0, 1, 2, 1]; ortho = [None, None, None, 1]
            if self.valence[i].item() is False:
                alpha, coeff = orthogonalize(
                    (alphas[i - 1], alpha),
                    (coeffs[i - 1], coeff),
                )

            alphas.append(alpha)
            coeffs.append(coeff)

        return alphas, coeffs

    @timing
    def unique_shell_pairs(self, ihelp: IndexHelper) -> tuple[Tensor, Tensor]:
        """Create a matrix of unique shell pairs.

        Parameters
        ----------
        ihelp : IndexHelper
            Helper class for indexing.

        Returns
        -------
        (Tensor, Tensor)
            Matrix of unique shell pairs and the number of unique shell pairs.

        Raises
        ------
        ValueError
            If the number of unique shell pairs does not match the theoretical one.
        """

        # offsets to avoid duplication on addition
        offset1 = 10000

        # convert unique shell indices to prime numbers for unique products
        sh2orb = ihelp.spread_shell_to_orbital(ihelp.shells_to_ushell)
        orbs = primes[sh2orb]
        orbs = orbs.unsqueeze(-2) * orbs.unsqueeze(-1)

        # extra offset along only one dimension to distinguish (n, m) and
        # (m, n) of the same orbital block (e.g. 1x3 sp and 3x1 ps block)
        sh2orb = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)
        orbs += sh2orb * offset1

        _, umap = torch.unique(orbs, return_inverse=True)

        # minimum number of unqiue shells pairs
        n_uang = self.angular.size(0)
        n_pairs = torch.sum(torch.arange(1, n_uang + 1))

        # add combinations due to extra offset along one dimension
        # non_zero = torch.count_nonzero(self.angular).item()
        # if non_zero != 0:
        #     n_pairs += torch.arange(n_uang - non_zero, n_uang).sum()

        #     # remove all symmetric blocks (pp, dd)
        #     remove = torch.bincount(self.angular)
        #     if self.angular[0].item() == 0:
        #         # remove count of s orbital
        #         remove = remove[1:]
        #     n_pairs -= torch.sum(remove - 1)

        # actual number of unique combinations of unique shells
        n_pairs_actual = torch.max(umap) + 1

        # check against theoretical number
        if n_pairs_actual < n_pairs:
            torch.set_printoptions(linewidth=200)
            print(umap)
            raise ValueError(
                f"Internal error: {n_pairs - n_pairs_actual} missing unique pairs."
            )

        return umap, n_pairs_actual


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

    # print(nsh, lsh, ortho)
    for ish in range(nsh):
        if ortho[ish] is not None:
            cgto[ish] = Cgto_Type(
                lsh[ish],
                *orthogonalize(
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
