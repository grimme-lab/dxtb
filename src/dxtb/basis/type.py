"""
Basis set class.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .._types import Literal, Tensor, TensorLike
from ..constants import PSE
from ..param import Param, get_elem_param, get_elem_pqn, get_elem_valence
from ..utils import real_pairs
from . import IndexHelper, orthogonalize, slater

# fmt: off
primes = torch.tensor(
    [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987]
)
"""The first 300 prime numbers"""
# fmt: on

angular2label = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
}


@dataclass
class CGTOBasis:
    angmom: int
    alphas: torch.Tensor  # (nbasis,)
    coeffs: torch.Tensor  # (nbasis,)
    normalized: bool = True

    def wfnormalize_(self) -> CGTOBasis:
        # will always be normalized already in dxtb because we have to also
        # include the orthonormalization of the H2s against the H1s
        return self


@dataclass
class AtomCGTOBasis:
    atomz: int | float | Tensor
    bases: list[CGTOBasis]
    pos: Tensor  # (ndim,)


class Basis(TensorLike):
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
        super().__init__(device, dtype)
        self.numbers = numbers
        self.meta = par.meta
        self.angular = angular

        self.ngauss = get_elem_param(
            numbers,
            par.element,
            "ngauss",
            device=self.device,
        )
        self.slater = get_elem_param(
            numbers,
            par.element,
            "slater",
            device=self.device,
            dtype=self.dtype,
        )
        self.pqn = get_elem_pqn(numbers, par.element, device=self.device)
        self.valence = get_elem_valence(numbers, par.element, device=self.device)

    def create_cgtos(self) -> tuple[list[Tensor], list[Tensor]]:
        """
        Create contracted Gaussian type orbitals from parametrization.

        Returns
        -------
        tuple[list[Tensor], list[Tensor]]
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

    def unique_shell_pairs(
        self,
        ihelp: IndexHelper,
        mask: Tensor | None = None,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
    ) -> tuple[Tensor, Tensor]:
        """
        Create a matrix of unique shell pairs.

        Parameters
        ----------
        ihelp : IndexHelper
            Helper class for indexing.
        mask : Tensor
            Mask for
        uplo : Literal["n", "N", "u", "U", "l", "L"]
            Whether the matrix of unique shell pairs should be create as a
            triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
            Defaults to `l` (lower triangular matrix).

        Returns
        -------
        (Tensor, Tensor)
            Matrix of unique shell pairs and the number of unique shell pairs.

        Raises
        ------
        ValueError
            The number of unique shell pairs does not match the theoretical one.
        """
        # NOTE: Zeros correspond to padding only if batched. They have
        # meaning for single runs and cannot be ignored in this case.
        sh2ush = ihelp.spread_shell_to_orbital(ihelp.shells_to_ushell)

        # FIXME: Maybe a bitwise operation is easier to understand? For now,
        # we convert unique shell indices to prime numbers to obtain unique
        # products upon multiplication (fundamental theorem of arithmetic).
        orbs = primes.to(self.device)[sh2ush]
        orbs = orbs.unsqueeze(-2) * orbs.unsqueeze(-1)
        sh2orb = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)

        # extra offset along only one dimension to distinguish (n, m) and
        # (m, n) of the same orbital block (e.g. 1x3 sp and 3x1 ps block)
        offset = 10000

        if ihelp.batched:
            orbs = torch.where(
                real_pairs(sh2ush),
                orbs + sh2orb.unsqueeze(-1) * offset,
                orbs.new_tensor(0),
            )
        else:
            orbs += sh2orb.unsqueeze(-1) * offset

        # catch systems with one single orbital (e.g. Helium)
        if orbs.size(-1) == 1:
            if ihelp.batched:
                raise NotImplementedError()
            return torch.tensor([[0]]), torch.tensor(1)

        if mask is not None:
            if orbs.shape != mask.shape:
                raise RuntimeError(
                    f"Shape of mask ({mask.shape}) and orbitals ({orbs.shape}) "
                    "does not match."
                )
            orbs = torch.where(mask, orbs, orbs.new_tensor(0))

        # fill remaining triangular matrix with dummy values
        # (must be negative to not interfere with the unique values)
        if uplo.casefold() == "l":
            umap = torch.unique(torch.tril(orbs), return_inverse=True)[1] - 1
        elif uplo.casefold() == "u":
            umap = torch.unique(torch.triu(orbs), return_inverse=True)[1] - 1
        elif uplo.casefold() == "n":
            umap = torch.unique(orbs, return_inverse=True)[1]

            # subtract 1 to mark masked values and avoid off-by-one-error
            if mask is not None and not mask.any():
                umap -= 1
        else:
            raise ValueError("Unknown option for `uplo`.")

        return umap, torch.max(umap) + 1

    def to_bse_gaussian94(
        self,
        ihelp: IndexHelper,
        save: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        with_header: bool = False,
    ) -> str:
        if self.numbers.ndim > 1:
            raise RuntimeError("Basis set printing does not work batched.")

        if len(self.numbers) == 0:
            raise ValueError("No atoms for basis set printout found.")

        if with_header is True:
            assert self.meta is not None

            l = 70 * "-"
            header = (
                f"!{l}\n"
                "! Basis Set Exchange\n"
                "! Version v0.9\n"
                "! https://www.basissetexchange.org\n"
                f"!{l}\n"
                f"!   Basis set: {self.meta.name}\n"
                f"! Description: Orthonormalized {self.meta.name} Basis\n"
                "!        Role: orbital\n"
                f"!     Version: {self.meta.version}\n"
                f"!{l}\n\n\n"
            )

        coeffs = []
        alphas = []
        s = 0
        txt = ""
        for i, number in enumerate(self.numbers.tolist()):
            if with_header is True:
                txt += header  # type: ignore

            txt += f"{PSE[number]}\n"

            shells = ihelp.shells_per_atom[i]
            for _ in range(shells):
                alpha, coeff = slater.to_gauss(
                    self.ngauss[s], self.pqn[s], self.angular[s], self.slater[s]
                )
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                l = angular2label[self.angular.tolist()[s]]
                txt += f"{l}    {len(alpha)}    1.00\n"
                for a, c in zip(alpha, coeff):
                    txt += f"      {a}      {c}\n"

                s += 1

            txt += "****\n"

            # SAVING
            if save is True:
                dpath = "./.database"
                fpath = f"{number:02d}.gaussian94"

                # Create the directory if it doesn't exist
                Path(dpath).mkdir(parents=True, exist_ok=True)

                # Create the file path
                file_path = Path(dpath) / fpath

                # Check if the file already exists
                if overwrite is False:
                    if file_path.exists():
                        print(
                            f"The file '{fpath}' already exists in the directory '{dpath}'."
                        )
                        continue

                # Save the file
                with open(file_path, "w", encoding="utf8") as file:
                    file.write(txt)

            if verbose is True:
                print(txt)

        return txt

    def to_bse_nwchem(
        self,
        ihelp: IndexHelper,
        save: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        with_header: bool = False,
    ) -> str:
        if self.numbers.ndim > 1:
            raise RuntimeError("Basis set printing does not work batched.")

        if len(self.numbers) == 0:
            raise ValueError("No atoms for basis set printout found.")

        if with_header is True:
            assert self.meta is not None

            l = 70 * "-"
            header = (
                f"!{l}\n"
                "! Basis Set Exchange\n"
                "! Version v0.9\n"
                "! https://www.basissetexchange.org\n"
                f"!{l}\n"
                f"!   Basis set: {self.meta.name}\n"
                f"! Description: Orthonormalized {self.meta.name} Basis\n"
                "!        Role: orbital\n"
                f"!     Version: {self.meta.version}\n"
                f"!{l}\n\n\n"
            )

        coeffs = []
        alphas = []
        s = 0
        fulltxt = ""
        for i, number in enumerate(self.numbers.tolist()):
            txt = ""
            if with_header is True:
                txt += header  # type: ignore

            shells = ihelp.shells_per_atom[i]
            for _ in range(shells):
                alpha, coeff = slater.to_gauss(
                    self.ngauss[s], self.pqn[s], self.angular[s], self.slater[s]
                )
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                l = angular2label[self.angular.tolist()[s]]
                txt += f"{PSE[number]}    {l}\n"
                for a, c in zip(alpha, coeff):
                    txt += f"      {a}      {c}\n"

                s += 1

            # SAVING
            if save is True:
                dpath = "./.database"
                fpath = f"{number:02d}.nwchem"

                # Create the directory if it doesn't exist
                Path(dpath).mkdir(parents=True, exist_ok=True)

                # Create the file path
                file_path = Path(dpath) / fpath

                # Check if the file already exists
                if overwrite is False:
                    if file_path.exists():
                        print(
                            f"The file '{fpath}' already exists in the directory '{dpath}'."
                        )
                        continue

                # Save the file
                with open(file_path, "w", encoding="utf8") as file:
                    file.write(txt)

            if verbose is True:
                print(txt)

            fulltxt += txt

        return fulltxt

    def create_dqc(self, positions: Tensor, ihelp: IndexHelper) -> list[AtomCGTOBasis]:
        if self.numbers.ndim > 1:
            raise NotImplementedError("Batch mode not implemented.")

        # collect final basis for each atom in list (same order as `numbers`)
        atombasis: list[AtomCGTOBasis] = []

        # tracking only required for orthogonalization
        coeffs = []
        alphas = []

        s = 0
        for i, number in enumerate(self.numbers):
            bases: list[CGTOBasis] = []
            shells = ihelp.shells_per_atom[i]

            for _ in range(shells):
                alpha, coeff = slater.to_gauss(
                    self.ngauss[s], self.pqn[s], ihelp.angular[s], self.slater[s]
                )

                # orthogonalize the H2s against the H1s
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                cgto = CGTOBasis(
                    angmom=ihelp.angular.tolist()[s],  # int!
                    alphas=alpha,
                    coeffs=coeff,
                    normalized=True,
                )
                bases.append(cgto)

                # increment
                s += 1

            atomcgtobasis = AtomCGTOBasis(
                atomz=number,
                bases=bases,
                pos=positions[i, :],
            )
            atombasis.append(atomcgtobasis)

        return atombasis
