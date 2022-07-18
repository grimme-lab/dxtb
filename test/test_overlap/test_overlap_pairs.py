import pytest
import torch

from xtbml.basis.type import Basis
from xtbml.basis import slater
from xtbml.integral import mmd
from xtbml.exceptions import IntegralTransformError
from xtbml.param.gfn1 import GFN1_XTB as par
from xtbml.utils import symbol2number


def test_overlap_h_c():
    """
    Compare against reference calculated with tblite-int H C 0,0,1.4 --bohr --method gfn1
    """

    numbers = symbol2number(["H", "C"])

    basis = Basis(numbers, par)
    h = basis.cgto.get("H")
    c = basis.cgto.get("C")

    vec = torch.tensor([0.0, 0.0, 1.4])

    ref = [
        torch.tensor([[+6.77212228e-01]]),
        torch.tensor([[+0.00000000e-00, +0.00000000e-00, -5.15340812e-01]]),
        torch.tensor([[+7.98499991e-02]]),
        torch.tensor([[+0.00000000e-00, +0.00000000e-00, -1.72674504e-01]]),
    ]

    for ish in h:
        for jsh in c:
            overlap = mmd.overlap(
                (ish.ang, jsh.ang),
                (ish.alpha[: ish.nprim], jsh.alpha[: jsh.nprim]),
                (ish.coeff[: ish.nprim], jsh.coeff[: jsh.nprim]),
                vec,
            )
            assert torch.allclose(
                overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
            )


def test_overlap_h_he():
    """
    Compare against reference calculated with tblite-int H He 0,0,1.7 --method gfn1 --bohr
    """

    numbers = symbol2number(["H", "He"])

    basis = Basis(numbers, par)
    h = basis.cgto.get("H")
    he = basis.cgto.get("He")

    vec = torch.tensor([0.0, 0.0, 1.7])

    ref = [
        torch.tensor([[3.67988976e-01]]),
        torch.tensor([[9.03638642e-02]]),
    ]

    for ish in h:
        for jsh in he:
            overlap = mmd.overlap(
                (ish.ang, jsh.ang),
                (ish.alpha[: ish.nprim], jsh.alpha[: jsh.nprim]),
                (ish.coeff[: ish.nprim], jsh.coeff[: jsh.nprim]),
                vec,
            )
            assert torch.allclose(
                overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
            )


def test_overlap_s_cl():
    """
    Compare against reference calculated with tblite-int S Cl 0,0,2.1 --method gfn1 --bohr
    """

    numbers = symbol2number(["S", "Cl"])

    basis = Basis(numbers, par)
    s = basis.cgto.get("S")
    cl = basis.cgto.get("Cl")

    vec = torch.tensor([0.0, 0.0, 2.1])

    ref = [
        torch.tensor([[4.21677786e-01]]),
        torch.tensor([[0.0, 0.0, -5.53353521e-01]]),
        torch.tensor([[+3.57843134e-01, 0.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[0.0], [0.0], [5.71415151e-01]]),
        torch.diag(
            torch.tensor(
                [4.48181566e-01, 4.48181566e-01, -2.31133305e-01],
            ),
        ),
        torch.tensor(
            [
                [0.0, 0.0, -0.07887177],
                [-0.49594098, 0.0, 0.0],
                [0.0, -0.49594098, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        ).T,
        torch.tensor([[0.40834189], [0.0], [0.0], [0.0], [0.0]]),
        torch.tensor(
            [
                [0.0, 0.49152805, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.49152805, 0.0, 0.0],
                [0.05231871, 0.0, 0.0, 0.0, 0.0],
            ],
        ).T,
        torch.diag(
            torch.tensor(
                [-0.03689737, -0.29504313, -0.29504313, 0.35847499, 0.35847499]
            )
        ),
    ]

    for ish in s:
        for jsh in cl:
            overlap = mmd.overlap(
                (ish.ang, jsh.ang),
                (ish.alpha[: ish.nprim], jsh.alpha[: jsh.nprim]),
                (ish.coeff[: ish.nprim], jsh.coeff[: jsh.nprim]),
                vec,
            )
            assert torch.allclose(
                overlap, ref.pop(0), rtol=1e-05, atol=1e-05, equal_nan=False
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_overlap_higher_orbitals(dtype):

    from xtbml.basis.type import _process_record
    from test_overlap.test_cgto_ortho_data import ref_data

    vec = torch.tensor([0.0, 0.0, 1.4], dtype=dtype)

    # arbitrary element
    ele = _process_record(par.element["Rn"])
    ish, jsh = ele[0], ele[1]
    ai = ish.alpha[: ish.nprim].type(dtype)
    ci = ish.coeff[: ish.nprim].type(dtype)
    aj = jsh.alpha[: jsh.nprim].type(dtype)
    cj = jsh.coeff[: jsh.nprim].type(dtype)

    # change momenta artifically for testing purposes
    for i in range(2):
        for j in range(2):
            ref = ref_data[(i, j)].type(dtype).T
            overlap = mmd.overlap(
                (i, j),
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

    from xtbml.basis.type import _process_record

    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element
    ele = _process_record(par.element["Rn"])
    ish, jsh = ele[0], ele[1]

    j = 5
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            mmd.overlap(
                (i, j),
                (ish.alpha[: ish.nprim], jsh.alpha[: jsh.nprim]),
                (ish.coeff[: ish.nprim], jsh.coeff[: jsh.nprim]),
                vec,
            )
    i = 5
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            mmd.overlap(
                (i, j),
                (ish.alpha[: ish.nprim], jsh.alpha[: jsh.nprim]),
                (ish.coeff[: ish.nprim], jsh.coeff[: jsh.nprim]),
                vec,
            )


@pytest.mark.parametrize("ng", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sto_ng_batch(ng, dtype):
    """
    Test symmetry of s integrals
    """
    n, l = 1, 0

    coeff, alpha = slater.to_gauss(ng, n, l, torch.tensor(1.0, dtype=dtype))
    coeff, alpha = coeff.type(dtype)[:ng], alpha.type(dtype)[:ng]
    angular = torch.tensor(l)
    vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    s = mmd.overlap((angular, angular), (alpha, alpha), (coeff, coeff), vec)

    assert torch.allclose(s[0, :], s[1, :])
    assert torch.allclose(s[0, :], s[2, :])
