"""Run tests for Hamiltonian."""

from __future__ import annotations
import pytest
import torch

from xtbml.basis.indexhelper import IndexHelper
from xtbml.exlibs.tbmalt import batch
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.param.util import get_element_angular
from xtbml.typing import Tensor
from xtbml.xtb.h0 import Hamiltonian

from .samples import samples


class Setup:
    """Setup class to define constants for test class."""

    atol: float = 1e-06
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

    @pytest.mark.parametrize("dtype", [torch.float])
    @pytest.mark.parametrize("name", ["SiH4", "PbH4-BiH3"])
    def test_overlap(self, dtype: torch.dtype, name: str) -> None:
        sample = samples[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["overlap"].type(dtype)
        # numbers = torch.tensor([2, 2, 1, 4])
        # positions = torch.zeros(4, 3)

        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o_old = h0.overlap()
        o = h0.overlap_new()
        for i in range(len(o)):
            for j in range(len(o)):
                if torch.abs(o[i, j] - ref[i, j]) > self.atol * 10:
                    print(
                        "new", i, j, o[i, j], ref[i, j], torch.abs(o[i, j] - ref[i, j])
                    )

                # if torch.abs(o_old[i, j] - ref[i, j]) > self.atol:
                #     print("old", i, j, o_old[i, j], ref[i, j])

        torch.set_printoptions(precision=1)
        # print(o)
        # print("")
        # print(ref)

        assert torch.allclose(o, o.mT, atol=self.atol)
        assert torch.allclose(o, ref, atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float])
    @pytest.mark.parametrize("name", ["SiH4_cn", "SiH4"])
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
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        if "cn" in name:
            cn = get_coordination_number(numbers, positions, exp_count)
        else:
            cn = None

        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        # print("numbers", numbers)
        # print("unique", unique)
        # print("angular", ihelp.angular)
        # print("unique_angular", ihelp.unique_angular)
        # print("atom_to_unique", ihelp.atom_to_unique)
        # print("shells_to_ushell", ihelp.shells_to_ushell)
        # print("shell_index", ihelp.shell_index)
        # print("shells_to_atom", ihelp.shells_to_atom)
        # print("shells_per_atom", ihelp.shells_per_atom)
        # print("orbital_index", ihelp.orbital_index)
        # print("orbitals_to_shell", ihelp.orbitals_to_shell)
        # print("orbitals_per_shell", ihelp.orbitals_per_shell)
        # print("\n")

        o_old = h0.overlap()
        o = h0.overlap_new()
        h = h0.build(o, cn=cn)

        assert torch.allclose(o, o.mT, atol=self.atol)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(h, ref, atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float])
    @pytest.mark.parametrize("name", ["PbH4-BiH3"])
    def test_h0_gfn1_medium(self, dtype: torch.dtype, name: str) -> None:
        """
        Compare against reference calculated with tblite
        """

        sample = samples[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        cn = get_coordination_number(numbers, positions, exp_count)
        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o_old = h0.overlap()
        o = h0.overlap_new()
        print(torch.allclose(o, o_old, atol=self.atol))
        print(o.shape)
        h = h0.build(o_old, cn=cn)
        print(h.shape)

        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                print(f"{i} {j} {h[i,j] - ref[i,j]}")

        assert torch.allclose(o, o.mT, atol=self.atol)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(h, ref, atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float])
    @pytest.mark.parametrize("name", ["LYS_xao"])
    def test_h0_gfn1_large(self, dtype: torch.dtype, name: str) -> None:
        """
        Compare against reference calculated with tblite
        """

        sample = samples[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        cn = get_coordination_number(numbers, positions, exp_count)
        ihelp = IndexHelper.from_numbers(numbers, get_element_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o_old = h0.overlap()
        o = h0.overlap_new()
        print(torch.allclose(o, o_old, atol=self.atol))
        print(o.shape)
        h = h0.build(o_old, cn=cn)
        print(h.shape)

        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                print(f"{i} {j} {h[i,j] - ref[i,j]}")

        assert torch.allclose(o, o.mT, atol=self.atol)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(h, ref, atol=self.atol)

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
