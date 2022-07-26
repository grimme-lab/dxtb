"""Run tests for Hamiltonian."""

from __future__ import annotations
import pytest
import torch
from xtbml.basis.indexhelper import IndexHelper
from xtbml.basis.ortho import orthogonalize
from xtbml.basis.slater import to_gauss
from xtbml.basis.type import Cgto_Type

from xtbml.exlibs.tbmalt import batch
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.param.util import get_elem_param, get_elem_valence, get_element_angular
from xtbml.typing import Tensor
from xtbml.xtb.h0 import Hamiltonian

from xtbml.integral import mmd
from xtbml.constants import ATOMIC_NUMBER, PSE
from xtbml.param import Element

from .samples import samples


class Setup:
    """Setup class to define constants for test class."""

    atol: float = 1e-05
    """Absolute tolerance for equality comparison in `torch.allclose`."""

    rtol: float = 1e-05
    """Relative tolerance for equality comparison in `torch.allclose`."""

    cutoff: Tensor = torch.tensor(30.0)
    """Cutoff for calculation of coordination number."""


class TestHamiltonian(Setup):
    """Testing the building of the Hamiltonian matrix."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    ##############
    #### GFN1 ####
    ##############

    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("name", ["C2H2"])
    def test_h0_gfn1(self, dtype: torch.dtype, name: str) -> None:
        """
        Compare against reference calculated with tblite-int:
        - H2: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1
        - H2_cn: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1 --cn 0.91396028097949444,0.91396028097949444
        - LiH: fpm run -- Li H 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1
        - HLi: fpm run -- H Li 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1
        - S2: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1 --cn 0.91396028097949444,0.91396028097949444
        - SiH4: tblite with "use dftd3_ncoord, only: get_coordination_number"
        - SiH4_cn: tblite with "use dftd3_ncoord, only: get_coordination_number"
        """

        sample = samples[name]
        numbers = sample["numbers"]
        unique = torch.unique(numbers)
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        if "cn" in name:
            cn = get_coordination_number(numbers, positions, exp_count)
        else:
            cn = None

        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        print("numbers", numbers)
        print("unique", unique)
        print("angular", ihelp.angular)
        print("unique_angular", ihelp.unique_angular)
        print("atom_to_unique", ihelp.atom_to_unique)
        print("shells_to_ushell", ihelp.shells_to_ushell)
        print("shell_index", ihelp.shell_index)
        print("shells_to_atom", ihelp.shells_to_atom)
        print("shells_per_atom", ihelp.shells_per_atom)
        print("orbital_index", ihelp.orbital_index)
        print("orbitals_to_shell", ihelp.orbitals_to_shell)
        print("orbitals_per_shell", ihelp.orbitals_per_shell)
        print("\n")

        # offset to avoid duplication on addition
        offset = 100

        orbs = ihelp.spread_shell_to_orbital(
            ihelp.shells_to_ushell
        ) + ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)
        orbs = orbs.unsqueeze(-2) + orbs.unsqueeze(-1)

        atoms = ihelp.spread_atom_to_orbital(ihelp.atom_to_unique)
        atoms = atoms.unsqueeze(-2) + atoms.unsqueeze(-1)

        u = atoms * offset + orbs
        unum, idx = torch.unique(u, return_inverse=True)
        print(unum)
        print(idx)

        # number of unqiue shells
        n_uangular = len(ihelp.unique_angular)

        # calculated number of unique combinations of unique shells
        n_unique_pairs = torch.max(idx) + 1

        # check against theoretical number
        n_uangular_comb = torch.sum(torch.arange(1, n_uangular + 1))
        if n_unique_pairs != n_uangular_comb:
            raise ValueError(
                f"Internal error: {n_uangular_comb- n_unique_pairs} missing unique pairs."
            )

        def get_pqn(
            numbers: Tensor,
            par_element: dict[str, Element],
            pad_val: int = -1,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ):
            key = "shells"

            shells = []
            for number in numbers:
                el = PSE.get(int(number.item()), "X")
                if el in par_element:
                    for shell in getattr(par_element[el], key):
                        shells.append(int(shell[0]))

                else:
                    shells.append(pad_val)

            return torch.tensor(shells, device=device, dtype=dtype)

        pqn = get_pqn(unique, par.element)
        print("\n\n\n")

        ngauss = get_elem_param(unique, par.element, "ngauss")
        slater = get_elem_param(unique, par.element, "slater", dtype=dtype)
        valence = get_elem_valence(unique, par.element, dtype=torch.uint8)

        cgto = []
        coeffs = []
        alphas = []
        angs = []

        for i in range(len(slater)):
            ret = to_gauss(ngauss[i], pqn[i], ihelp.unique_angular[i], slater[i])
            cgtoi = Cgto_Type(ihelp.unique_angular[i], *ret)

            if valence[i].item() == 0:
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
            angs.append(cgtoi.ang)

        # a little hacky...
        al = batch.index(
            batch.index(batch.pack(alphas, value=0), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )
        c = batch.index(
            batch.index(batch.pack(coeffs, value=0), ihelp.shells_to_ushell),
            ihelp.orbitals_to_shell,
        )

        ################################################

        print(positions)
        positions = batch.index(positions, ihelp.shells_to_atom)
        positions = batch.index(positions, ihelp.orbitals_to_shell)
        print(positions)

        vec = positions.unsqueeze(-2) - positions

        a = torch.zeros((len(orbs), len(orbs)))
        a_ = a[None, None, :, None, :]
        print(a_.shape)
        a_row_chunks = torch.cat(torch.chunk(a_, 3, dim=2), dim=1)
        print(a_row_chunks.shape)
        a_col_chunks = torch.cat(torch.chunk(a_row_chunks, 3, dim=4), dim=3)
        a_chunks = a_col_chunks.reshape(1, 3, 3, 4, 4)

        indx = torch.tensor([[0, 2, 0], [0, 2, 4], [0, 4, 0]])
        indx_ = indx.clone().float()
        indx_[:, 1:] /= 2
        indx_ = indx_.long()
        print(a_chunks)
        return

        ovlp = torch.zeros((len(orbs), len(orbs)))
        for i in range(n_unique_pairs):
            print(i)
            i = 9
            pairs = (idx == i).nonzero(as_tuple=False)
            coeff = c[pairs[0]]
            alpha = al[pairs[0]]
            ang = ihelp.spread_shell_to_orbital(ihelp.angular)[pairs[0]]
            alpha_tuple = (batch.deflate(alpha[0]), batch.deflate(alpha[1]))
            coeff_tuple = (batch.deflate(coeff[0]), batch.deflate(coeff[1]))
            print(ang[0], ang[1], pairs[0])
            print(pairs)
            # print(ihelp.reduce_orbital_to_atom(pairs))
            print(ihelp.spread_shell_to_orbital(ihelp.angular))
            # print(alpha_tuple[0], alpha_tuple[1])
            # print(coeff_tuple[0], coeff_tuple[1])

            vec = positions[pairs][:, 0, :] - positions[pairs][:, 1, :]
            stmp = mmd.overlap((ang[0], ang[1]), alpha_tuple, coeff_tuple, -vec)
            print(stmp)
            # ovlp[pairs[:, 0], pairs[:, 1]] = stmp.flatten()
            # print(ovlp)

            # print(positions[pairs][:, 0, :] - positions[pairs][:, 1, :])
            # print(positions[pairs])
            # print(positions[pairs[0]])
            break

        print("\n")
        o = h0.overlap()
        # print(o)
        h = h0.build(o, cn=cn)

        assert torch.all(h.transpose(-2, -1) == h)
        # assert torch.allclose(h, ref, rtol=self.rtol, atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("name1", ["H2", "LiH", "S2", "SiH4"])
    @pytest.mark.parametrize("name2", ["H2", "LiH", "S2", "SiH4"])
    def stest_h0_gfn1_batch(self, dtype: torch.dtype, name1: str, name2: str) -> None:
        """Batched version."""

        sample1, sample2 = samples[name1], samples[name2]

        numbers = batch.pack(
            (
                sample1["numbers"],
                sample2["numbers"],
            )
        )
        positions = batch.pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )
        ref = batch.pack(
            (
                sample1["h0"].type(dtype),
                sample2["h0"].type(dtype),
            ),
        )

        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)
        o = h0.overlap()
        h = h0.build(o)

        assert torch.allclose(h, ref, atol=self.atol, rtol=self.rtol)

    ##############
    #### GFN2 ####
    ##############

    # def test_hamiltonian_h2_gfn2(self) -> None:
    #     nao = 2
    #     ref_hamiltonian = torch.tensor([
    #         -3.91986875628795E-1, -4.69784163992013E-1,
    #         -4.69784163992013E-1, -3.91986875628795E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_H2"], ref_hamiltonian)

    # def test_hamiltonian_lih_gfn2(self) -> None:
    #     nao = 5
    #     ref_hamiltonian = torch.tensor([
    #         -1.85652586923456E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.04060196214555E-1, 0.00000000000000E+0,
    #         -7.93540972812401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.93540972812401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.93540972812401E-2, -2.64332062163992E-1, -2.04060196214555E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.64332062163992E-1,
    #         -3.91761139212137E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_LiH"], ref_hamiltonian)

    # def test_hamiltonian_s2_gfn2(self) -> None:
    #     nao = 18
    #     ref_hamiltonian = torch.tensor([
    #         -7.35145147501899E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.92782969898654E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.36427116435023E-1, -2.05951870741313E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.17765757158496E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -9.33756556185781E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.17765757158496E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -9.33756556185781E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.20176381592757E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.17765757158496E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.36427116435023E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.58607478516679E-1, -1.23733824351312E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.05951870741313E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.23733824351312E-1, -9.40352474745915E-3, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.20176381592757E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 2.45142819855746E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.27200169863736E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.45142819855746E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.00344447956584E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.00344447956584E-2,
    #         -1.92782969898654E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.36427116435023E-1, -2.05951870741313E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -7.35145147501899E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -9.33756556185781E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.17765757158496E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -9.33756556185781E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.20176381592757E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.17765757158496E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.36427116435023E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.58607478516679E-1, 1.23733824351312E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.17765757158496E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.05951870741313E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -1.23733824351312E-1, -9.40352474745915E-3, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 1.20176381592757E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 2.45142819855746E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.20176381592757E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         2.45142819855746E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.27200169863736E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.00344447956584E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.27200169863736E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.00344447956584E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -2.27200169863736E-2
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_S2"], ref_hamiltonian)

    # def test_hamiltonian_sih4_gfn2(self) -> None:
    #     nao = 13
    #     ref_hamiltonian = torch.tensor([
    #         -5.52420992289823E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -3.36004309624475E-1, -3.36004309624475E-1, -3.36004309624475E-1,
    #         -3.36004309624475E-1, 0.00000000000000E+0, -2.35769689453471E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -1.53874693770200E-1, 1.53874693770200E-1,
    #         -1.53874693770200E-1, 1.53874693770200E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, -2.35769689453471E-1, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -1.53874693770200E-1,
    #         1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -2.35769689453471E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         -1.53874693770200E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.13801957898401E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.13801957898401E-2, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -1.23912378305726E-1, -1.23912378305726E-1, 1.23912378305726E-1,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         -4.13801957898401E-2, 0.00000000000000E+0, 0.00000000000000E+0,
    #         1.23912378305726E-1, -1.23912378305726E-1, 1.23912378305726E-1,
    #         -1.23912378305726E-1, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, -4.13801957898401E-2,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, 0.00000000000000E+0, 0.00000000000000E+0,
    #         0.00000000000000E+0, -4.13801957898401E-2, -1.23912378305726E-1,
    #         -1.23912378305726E-1, 1.23912378305726E-1, 1.23912378305726E-1,
    #         -3.36004309624475E-1, -1.53874693770200E-1, -1.53874693770200E-1,
    #         1.53874693770200E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         1.23912378305726E-1, 0.00000000000000E+0, -1.23912378305726E-1,
    #         -3.91823578951118E-1, -4.31486716382575E-2, -4.31486716382575E-2,
    #         -4.31486716382575E-2, -3.36004309624475E-1, 1.53874693770200E-1,
    #         1.53874693770200E-1, 1.53874693770200E-1, 0.00000000000000E+0,
    #         -1.23912378305726E-1, -1.23912378305726E-1, 0.00000000000000E+0,
    #         -1.23912378305726E-1, -4.31486716382575E-2, -3.91823578951118E-1,
    #         -4.31486716382575E-2, -4.31486716382575E-2, -3.36004309624475E-1,
    #         -1.53874693770200E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         0.00000000000000E+0, -1.23912378305726E-1, 1.23912378305726E-1,
    #         0.00000000000000E+0, 1.23912378305726E-1, -4.31486716382575E-2,
    #         -4.31486716382575E-2, -3.91823578951118E-1, -4.31486716382575E-2,
    #         -3.36004309624475E-1, 1.53874693770200E-1, -1.53874693770200E-1,
    #         -1.53874693770200E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -1.23912378305726E-1, 0.00000000000000E+0, 1.23912378305726E-1,
    #         -4.31486716382575E-2, -4.31486716382575E-2, -4.31486716382575E-2,
    #         -3.91823578951118E-1
    #     ]).reshape(nao, nao)

    #     self.base_test(data["MB16_43_SiH4"], ref_hamiltonian)
