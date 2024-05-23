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
Basis: Main Class
=================

Main basis set class for creating the contracted Gaussian type orbitals (CGTOs)
from the parametrization. The basis set can also be printed in various formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import tensor_to_numpy
from tad_mctc.data import pse
from tad_mctc.exceptions import DtypeError

from dxtb._src.param import Param, get_elem_param, get_elem_pqn, get_elem_valence
from dxtb._src.typing import Literal, Self, Tensor, TensorLike, override

from .indexhelper import IndexHelper
from .ortho import orthogonalize
from .slater import slater_to_gauss

if TYPE_CHECKING:
    from dxtb._src.exlibs import libcint
del TYPE_CHECKING

__all__ = ["Basis"]


# fmt: off
primes = torch.tensor(
    [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987]
)
"""The first 300 prime numbers."""
# fmt: on

angular2label = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
}


class Basis(TensorLike):
    """Atomic orbital basis set."""

    ngauss: Tensor
    """Number of Gaussians used in expansion from Slater orbital."""

    slater: Tensor
    """Exponent of Slater function."""

    pqn: Tensor
    """Principal quantum number of each shell"""

    valence: Tensor
    """Whether the shell is part of the valence shell."""

    __slots__ = [
        "numbers",
        "unique",
        "meta",
        "ihelp",
        "ngauss",
        "slater",
        "pqn",
        "valence",
    ]

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
        self.unique: Tensor = torch.unique(numbers)
        self.meta = par.meta
        self.ihelp = ihelp

        self.ngauss = get_elem_param(
            self.unique, par.element, "ngauss", device=self.device, dtype=torch.uint8
        )
        self.slater = get_elem_param(self.unique, par.element, "slater", **self.dd)
        self.pqn = get_elem_pqn(
            self.unique, par.element, device=self.device, dtype=torch.uint8
        )
        self.valence = get_elem_valence(self.unique, par.element, device=self.device)

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
        for i in range(self.ihelp.unique_angular.size(0)):
            alpha, coeff = slater_to_gauss(
                self.ngauss[i],
                self.pqn[i],
                self.ihelp.unique_angular[i],
                self.slater[i],
            )

            # NOTE:
            # This only works for GFN1 with H being the only element
            # with a non-valence shell. Otherwise the generation of
            # self.valence is not correct.
            # The correct way would be a map to show which orbital needs
            # to be orthogonalized w.r.t. another one.
            # Example: Si with 3s, 3p, 3d, 4p
            # angular = [0, 1, 2, 1]; ortho = [None, None, None, 1]
            # However, it is probably only ever needed in GFN1. Other methods
            # do not specifically orthogonalize certain basis functions.
            if self.meta is not None and self.meta.name is not None:
                if "gfn1" in self.meta.name.casefold():
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
        mask: Tensor | None = None,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
    ) -> tuple[Tensor, Tensor]:
        """
        Create a matrix of unique shell pairs.

        Parameters
        ----------
        mask : Tensor
            Mask for
        uplo : Literal["n", "N", "u", "U", "l", "L"]
            Whether the matrix of unique shell pairs should be create as a
            triangular matrix (``l``: lower, ``u``: upper) or full matrix
            (``n``). Defaults to ``l`` (lower triangular matrix).

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
        sh2ush = self.ihelp.spread_shell_to_orbital(self.ihelp.shells_to_ushell)

        # NOTE: Maybe a bitwise operation is easier to understand? For now,
        # we convert unique shell indices to prime numbers to obtain unique
        # products upon multiplication (fundamental theorem of arithmetic).
        orbs = primes.to(self.device)[sh2ush]
        orbs = orbs.unsqueeze(-2) * orbs.unsqueeze(-1)
        sh2orb = self.ihelp.spread_shell_to_orbital(self.ihelp.orbitals_per_shell)

        # extra offset along only one dimension to distinguish (n, m) and
        # (m, n) of the same orbital block (e.g. 1x3 sp and 3x1 ps block)
        offset = 10000

        if self.ihelp.batch_mode > 0:
            orbs = torch.where(
                real_pairs(sh2ush, mask_diagonal=False),
                orbs + sh2orb.unsqueeze(-1) * offset,
                torch.tensor(0, device=self.ihelp.device),
            )
        else:
            orbs += sh2orb.unsqueeze(-1) * offset

        # catch systems with one single orbital (e.g. Helium)
        if orbs.size(-1) == 1:
            if self.ihelp.batch_mode > 0:
                raise NotImplementedError()
            return torch.tensor([[0]]), torch.tensor(1)

        if mask is not None:
            if orbs.shape != mask.shape:
                raise RuntimeError(
                    f"Shape of mask ({mask.shape}) and orbitals ({orbs.shape}) "
                    "does not match."
                )
            orbs = torch.where(mask, orbs, torch.tensor(0, device=orbs.device))

        # fill remaining triangular matrix with dummy values
        # (must be negative to not interfere with the unique values)
        if uplo.casefold() == "l":
            umap = torch.unique(torch.tril(orbs), return_inverse=True)[1] - 1
        elif uplo.casefold() == "u":
            umap = torch.unique(torch.triu(orbs), return_inverse=True)[1] - 1
        elif uplo.casefold() == "n":
            umap = torch.unique(orbs, return_inverse=True)[1]

            # subtract 1 to mark masked values and avoid off-by-one-error
            # (mask is active if it contains at least one ``False`` value)
            if mask is not None and (~mask).any():
                umap -= 1
        else:
            raise ValueError("Unknown option for `uplo`.")

        return umap, torch.max(umap) + 1

    def to_bse(
        self,
        qcformat: Literal["gaussian94", "nwchem"] = "nwchem",
        save: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        with_header: bool = False,
    ) -> str:
        """
        Convert the basis set to a format suitable for basis set exchange.

        Parameters
        ----------
        qcformat : Literal["gaussian94", "nwchem"], optional
            Format of the basis set. Defaults to ``"nwchem"``.
        save : bool, optional
            Whether to save the basis set to a file. Defaults to ``False``.
        overwrite : bool, optional
            Whether to overwrite existing files. Defaults to ``False``.
        verbose : bool, optional
            Whether to print the basis set to the console.
            Defaults to ``False``.
        with_header : bool, optional
            Whether to include the header in the basis set.
            Defaults to ``False``.

        Returns
        -------
        str
            Basis set in the specified format.

        Raises
        ------
        RuntimeError
            If no meta data is found in the parametrization, or if the meta
            data is incomplete (name missing), or if the basis set is batched.
        ValueError
            If no atoms for basis set printout are found, or if the basis set
            format is not supported.
        """
        if self.meta is None:
            raise RuntimeError("No meta data found in the parametrization.")

        if self.unique.ndim > 1:
            raise RuntimeError("Basis set printing does not work batched.")

        if len(self.unique) == 0:
            raise ValueError("No atoms for basis set printout found.")

        allowed_formats = ("gaussian94", "nwchem")
        if qcformat not in allowed_formats:
            raise ValueError(
                f"Basis set format '{qcformat}' not supported. "
                f"Available options are: {allowed_formats}."
            )

        if with_header is True:
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
        for i, number in enumerate(self.unique.tolist()):
            txt = ""
            if with_header is True:
                txt += header  # type: ignore

            if qcformat == "gaussian94":
                txt += f"{pse.Z2S[number]}\n"

            shells = self.ihelp.shells_per_atom[i]
            for _ in range(shells):
                alpha, coeff = slater_to_gauss(
                    self.ngauss[s], self.pqn[s], self.ihelp.angular[s], self.slater[s]
                )
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                l = angular2label[self.ihelp.angular.tolist()[s]]
                if qcformat == "gaussian94":
                    txt += f"{l}    {len(alpha)}    1.00\n"
                elif qcformat == "nwchem":
                    txt += f"{pse.Z2S[number]}    {l}\n"

                # write exponents and coefficients
                for a, c in zip(alpha, coeff):
                    txt += f"      {a}      {c}\n"

                s += 1

            # final separator in gaussian94 format
            if qcformat == "gaussian94":
                txt += "****\n"

            # always save to src/dxtb/mol/external/basis
            if save is True:
                if self.meta.name is None:
                    raise RuntimeError("Meta data incomplete (name missing).")

                # Create the directory if it doesn't exist
                target = f"mol/external/basis/{self.meta.name.casefold()}"
                dpath = Path(__file__).parents[1] / target
                Path(dpath).mkdir(parents=True, exist_ok=True)

                # Create the file path
                fpath = f"{number:02d}.nwchem"
                file_path = Path(dpath) / fpath

                # Check if the file already exists
                if overwrite is False:
                    if file_path.exists():
                        print(
                            f"The file '{fpath}' already exists in the "
                            f"directory '{dpath}'. It will not be overwritten."
                        )
                        continue

                # Save the file
                with open(file_path, "w", encoding="utf8") as file:
                    file.write(txt)

            if verbose is True:
                print(txt)

            fulltxt += txt

        return fulltxt

    def create_libcint(
        self, positions: Tensor, mask: Tensor | None = None
    ) -> list[libcint.AtomCGTOBasis] | list[list[libcint.AtomCGTOBasis]]:
        """
        Create the basis set required for `libcint`.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        mask : Tensor | None, optional
            Mask for positions to make batched computations easier. The overlap
            does not work in a batched fashion. Hence, we loop over the batch
            dimension and must remove the padding. Defaults to ``None``, i.e.,
            :func:`tad_mctc.batch.deflate` is used.

        Returns
        -------
        list[libcint.AtomCGTOBasis] | list[list[libcint.AtomCGTOBasis]]
            List of CGTOs.

        Raises
        ------
        NotImplementedError
            If batch mode is requested (checked through dimensions of numbers).
        """
        if self.unique.ndim > 1:
            raise NotImplementedError("Batch mode not implemented.")

        # pylint: disable=import-outside-toplevel
        from dxtb._src.exlibs import libcint

        # tracking only required for orthogonalization
        coeffs = []
        alphas = []

        s = 0
        for i in range(self.unique.size(0)):
            bases: list[libcint.CGTOBasis] = []
            shells = self.ihelp.ushells_per_unique[i]

            if shells == 0:
                s += 1
                zero = torch.tensor(0.0, **self.dd)
                alphas.append(zero)
                coeffs.append(zero)
                continue

            for _ in range(shells):
                alpha, coeff = slater_to_gauss(
                    self.ngauss[s],
                    self.pqn[s],
                    self.ihelp.unique_angular[s],
                    self.slater[s],
                )

                # orthogonalize the H2s against the H1s
                if self.valence[s].item() is False:
                    alpha, coeff = orthogonalize(
                        (alphas[s - 1], alpha),
                        (coeffs[s - 1], coeff),
                    )
                alphas.append(alpha)
                coeffs.append(coeff)

                # increment
                s += 1

        ##########
        # SINGLE #
        ##########

        if self.ihelp.batch_mode == 0:
            # reset counter
            s = 0

            # collect final basis for each atom in list (same order as `numbers`)
            atombasis: list[libcint.AtomCGTOBasis] = []

            for i, num in enumerate(self.numbers):
                bases: list[libcint.CGTOBasis] = []

                for _ in range(self.ihelp.shells_per_atom[i]):
                    idx = self.ihelp.shells_to_ushell[s]

                    cgto = libcint.CGTOBasis(
                        angmom=int(self.ihelp.angular[s]),  # int!
                        alphas=alphas[idx],
                        coeffs=coeffs[idx],
                        normalized=True,
                    )
                    bases.append(cgto)

                    # increment
                    s += 1

                atomcgtobasis = libcint.AtomCGTOBasis(
                    atomz=num,
                    bases=bases,
                    pos=positions[i, :],
                )
                atombasis.append(atomcgtobasis)

            return atombasis

        ###########
        # BATCHED #
        ###########

        # collection for batch
        b: list[list[libcint.AtomCGTOBasis]] = []

        for _batch in range(self.numbers.shape[0]):
            # reset counter
            s = 0

            # collect final basis for each atom in list
            atombasis: list[libcint.AtomCGTOBasis] = []

            shell_idxs = self.ihelp.shells_to_ushell[_batch]
            shells_per_atom = self.ihelp.shells_per_atom[_batch]
            angular = tensor_to_numpy(self.ihelp.angular[_batch])
            pos = positions[_batch]

            for i, num in enumerate(self.numbers[_batch]):
                if num == 0:
                    continue

                # CGTOs
                bases: list[libcint.CGTOBasis] = []
                for _ in range(shells_per_atom[i]):
                    idx = shell_idxs[s]

                    # FIXME: Should probably be some kind of mask to get rid of
                    # padding? But "s" only runs over the actual shells, so it
                    # should be fine.
                    # However, batched mode not working for functorch AD.
                    cgto = libcint.CGTOBasis(
                        angmom=angular[s],  # int!
                        alphas=alphas[idx],
                        coeffs=coeffs[idx],
                        normalized=True,
                    )
                    bases.append(cgto)

                    # increment
                    s += 1

                # POSITIONS
                if self.ihelp.batch_mode == 1:
                    if mask is not None:
                        m = mask[_batch]
                        _pos = torch.masked_select(pos, m).reshape((-1, 3))
                    else:
                        # pylint: disable=import-outside-toplevel
                        from tad_mctc.batch import deflate

                        _pos = deflate(pos, value=float("nan"))
                elif self.ihelp.batch_mode == 2:
                    _pos = pos

                atomcgtobasis = libcint.AtomCGTOBasis(
                    atomz=num,
                    bases=bases,
                    pos=_pos[i, :],
                )
                atombasis.append(atomcgtobasis)

            b.append(atombasis)

        return b

    @override
    def type(self, dtype: torch.dtype) -> Self:
        """
        Returns a copy of the class instance with specified floating point type.

        This method overrides the usual approach because the
        :class:`~dxtb.Calculator`'s arguments and slots differ significantly.
        Hence, it is not practical to instantiate a new copy.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        Self
            A copy of the class instance with the specified dtype.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        DtypeError
            If the specified dtype is not allowed.
        """

        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise DtypeError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        self.numbers = self.numbers.type(dtype)
        self.unique = self.unique.type(dtype)
        self.slater = self.slater.type(dtype)

        # hard override of the dtype in TensorLike
        self.override_dtype(dtype)

        return self

    @override
    def to(self, device: torch.device) -> Self:
        """
        Returns a copy of the class instance on the specified device.

        This method overrides the usual approach because the
        :class:`~dxtb.Calculator`'s arguments and slots differ significantly.
        Hence, it is not practical to instantiate a new copy.

        Parameters
        ----------
        device : torch.device
            Device to store the tensor on.

        Returns
        -------
        Self
            A copy of the class instance on the specified device.

        Raises
        ------
        RuntimeError
            If the ``__slots__`` attribute is not set in the class.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting ``__slots__`` in the "
                f"'{self.__class__.__name__}' class."
            )

        self.numbers = self.numbers.to(device)
        self.unique = self.unique.to(device)
        self.ihelp = self.ihelp.to(device)
        self.slater = self.slater.to(device)
        self.valence = self.valence.to(device)
        self.ngauss = self.ngauss.to(device)
        self.pqn = self.pqn.to(device)

        # hard override of the dtype in TensorLike
        self.override_device(device)

        return self
