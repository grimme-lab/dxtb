"""Run tests for overlap."""

import numpy as np
import pytest
import torch

from dxtb.basis import Basis, IndexHelper, slater
from dxtb.integral import Overlap, mmd
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import IntegralTransformError, batch

from ..utils import load_from_npz
from .samples import samples

ref_overlap = np.load("test/test_overlap/overlap.npz")


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("name", ["HC", "HHe", "SCl"])
def test_single(dtype: torch.dtype, name: str):
    """
    Compare against reference calculated with tblite-int:
    - HC: fpm run -- H C 0,0,1.4 --bohr --method gfn1
    - HHe: fpm run -- H He 0,0,1.7 --bohr --method gfn1
    - SCl: fpm run -- S Cl 0,0,2.1 --bohr --method gfn1
    """
    dd = {"dtype": dtype}
    tol = 1e-05

    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    ref = load_from_npz(ref_overlap, name, dtype)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)

    assert pytest.approx(s, rel=tol, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("name1", ["C", "HC", "HHe", "SCl"])
@pytest.mark.parametrize("name2", ["C", "HC", "HHe", "SCl"])
def test_batch(dtype: torch.dtype, name1: str, name2: str) -> None:
    """Batched version."""
    dd = {"dtype": dtype}
    tol = 1e-05

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
            load_from_npz(ref_overlap, name1, dtype),
            load_from_npz(ref_overlap, name2, dtype),
        ),
    )

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    overlap = Overlap(numbers, par, ihelp, **dd)
    s = overlap.build(positions)

    assert pytest.approx(s, abs=tol) == s.mT
    assert pytest.approx(s, abs=tol) == ref


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_overlap_higher_orbitals(dtype: torch.dtype):

    # pylint: disable=import-outside-toplevel
    from .test_cgto_ortho_data import ref_data

    vec = torch.tensor([0.0, 0.0, 1.4], dtype=dtype)

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, get_elem_angular(par.element))
    bas = Basis(number, par, ihelp.unique_angular)
    alpha, coeff = bas.create_cgtos()

    ai = alpha[0]
    ci = coeff[0]
    aj = alpha[1]
    cj = coeff[1]

    # change momenta artifically for testing purposes
    for i in range(2):
        for j in range(2):
            ref = ref_data[f"{i}-{j}"].type(dtype).T
            overlap = mmd.overlap(
                (torch.tensor(i), torch.tensor(j)),
                (ai, aj),
                (ci, cj),
                vec,
            )

            assert torch.allclose(
                overlap,
                ref,
                rtol=1e-05,
                atol=1e-03,
                equal_nan=False,
            )


def test_overlap_higher_orbital_fail():
    """No higher orbitals than 4 allowed."""

    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, get_elem_angular(par.element))
    bas = Basis(number, par, ihelp.unique_angular)
    alpha, coeff = bas.create_cgtos()

    j = torch.tensor(5)
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            mmd.overlap(
                (torch.tensor(i), j),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
    i = torch.tensor(5)
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            mmd.overlap(
                (i, torch.tensor(j)),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_sto_ng_batch(ng: int, dtype: torch.dtype):
    """
    Test symmetry of s integrals
    """
    n, l = torch.tensor(1), torch.tensor(0)
    ng_ = torch.tensor(ng)

    coeff, alpha = slater.to_gauss(ng_, n, l, torch.tensor(1.0, dtype=dtype))
    coeff, alpha = coeff.type(dtype)[:ng_], alpha.type(dtype)[:ng_]
    vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    s = mmd.overlap((l, l), (alpha, alpha), (coeff, coeff), vec)

    assert torch.allclose(s[0, :], s[1, :])
    assert torch.allclose(s[0, :], s[2, :])
