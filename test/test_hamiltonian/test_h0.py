"""Run tests for Hamiltonian."""

from __future__ import annotations
import pytest
import torch

from xtbml.basis.indexhelper import IndexHelper
from xtbml.exlibs.tbmalt import batch
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.param.util import get_elem_angular
from xtbml.typing import Tensor
from xtbml.utils import combinations as combis
from xtbml.xtb.h0 import Hamiltonian

from .samples import samples

small = ["C", "Rn", "H2", "H2_nocn", "LiH", "HLi", "S2", "SiH4", "SiH4_nocn"]
large = ["PbH4-BiH3", "LYS_xao"]


class Setup:
    """Setup class to define constants for test class."""

    atol: float = 1e-06
    """Absolute tolerance for equality comparison in `torch.allclose`."""

    rtol: float = 1e-05
    """Relative tolerance for equality comparison in `torch.allclose`."""

    cutoff: Tensor = torch.tensor(30.0)
    """Cutoff for calculation of coordination number."""


class TestHamiltonianGFN1(Setup):
    """Testing the building of the Hamiltonian matrix."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name", small)
    def test_h0(self, dtype: torch.dtype, name: str) -> None:
        """
        Compare against reference calculated with tblite-int:
        - H2: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1 --cn 0.91396028097949444,0.91396028097949444
        - H2_nocn: fpm run -- H H 0,0,1.4050586229538 --bohr --hamiltonian --method gfn1
        - LiH: fpm run -- Li H 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1 --cn 0.98684772035550494,0.98684772035550494
        - HLi: fpm run -- H Li 0,0,3.0159348779447 --bohr --hamiltonian --method gfn1 --cn 0.98684772035550494,0.98684772035550494
        - S2: fpm run -- S S 0,0,3.60562542949258 --bohr --hamiltonian --method gfn1 --cn 0.99889747382180494,0.99889747382180494
        - SiH4: tblite with "use dftd3_ncoord, only: get_coordination_number"
        - SiH4_nocn: tblite
        """

        sample = samples[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        if "nocn" in name:
            cn = None
        else:
            cn = get_coordination_number(numbers, positions, exp_count)

        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o = h0.overlap()
        assert torch.allclose(o, o.mT, atol=self.atol)

        h = h0.build(o, cn=cn)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(h, ref, atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name1", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
    @pytest.mark.parametrize("name2", ["C", "Rn", "H2", "LiH", "S2", "SiH4"])
    def test_h0_batch(self, dtype: torch.dtype, name1: str, name2: str) -> None:
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

        cn = get_coordination_number(numbers, positions, exp_count)
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o = h0.overlap()
        assert torch.allclose(o, o.mT, atol=self.atol)

        h = h0.build(o, cn)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(h, ref, atol=self.atol, rtol=self.rtol)

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name", large)
    def test_h0_large(self, dtype: torch.dtype, name: str) -> None:
        """Compare against reference calculated with tblite"""

        sample = samples[name]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        ref = sample["h0"].type(dtype)

        cn = get_coordination_number(numbers, positions, exp_count)
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o = h0.overlap()
        assert torch.allclose(o, o.mT, atol=self.atol)

        h = h0.build(o, cn=cn)
        assert torch.allclose(h, h.mT, atol=self.atol)
        assert torch.allclose(combis(h), combis(ref), atol=self.atol)

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("name1", large)
    @pytest.mark.parametrize("name2", large)
    def test_h0_large_batch(self, dtype: torch.dtype, name1: str, name2: str) -> None:
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

        cn = get_coordination_number(numbers, positions, exp_count)
        ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
        h0 = Hamiltonian(numbers, positions, par, ihelp)

        o = h0.overlap()
        assert torch.allclose(o, o.mT, atol=self.atol)

        h = h0.build(o, cn)
        assert torch.allclose(h, h.mT, atol=self.atol)

        for _batch in range(numbers.shape[0]):
            assert torch.allclose(
                combis(h[_batch]), combis(ref[_batch]), atol=self.atol
            )
