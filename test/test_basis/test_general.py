"""
Run tests for overlap of diatomic systems.
References calculated with tblite 0.3.0.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.basis import Basis, IndexHelper, slater
from dxtb.integral import overlap_gto
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import (
    CGTOAzimuthalQuantumNumberError,
    CGTOPrimitivesError,
    CGTOPrincipalQuantumNumberError,
    CGTOQuantumNumberError,
    CGTOSlaterExponentsError,
    IntegralTransformError,
)


def test_fail_number_primitives() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOPrimitivesError):
        slater.to_gauss(torch.tensor(7), n, l, torch.tensor(1.2))


def test_fail_slater_exponent() -> None:
    # principal and azimuthal quantum number of 1s-orbital
    n, l = torch.tensor(1), torch.tensor(0)

    with pytest.raises(CGTOSlaterExponentsError):
        slater.to_gauss(torch.tensor(6), n, l, torch.tensor(-1.2))


def test_fail_max_principal() -> None:
    # principal and azimuthal quantum number of 7s-orbital
    n, l = torch.tensor(7), torch.tensor(0)

    with pytest.raises(CGTOPrincipalQuantumNumberError):
        slater.to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital() -> None:
    # principal and azimuthal quantum number of 5h-orbital
    n, l = torch.tensor(5), torch.tensor(5)

    with pytest.raises(CGTOAzimuthalQuantumNumberError):
        slater.to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_quantum_number() -> None:
    # principal and azimuthal quantum number of 2f-orbital
    n, l = torch.tensor(2), torch.tensor(3)

    with pytest.raises(CGTOQuantumNumberError):
        slater.to_gauss(torch.tensor(6), n, l, torch.tensor(1.2))


def test_fail_higher_orbital_trafo():
    """No higher orbitals than d-orbitals allowed."""
    vec = torch.tensor([0.0, 0.0, 1.4])

    # arbitrary element (Rn)
    number = torch.tensor([86])

    ihelp = IndexHelper.from_numbers(number, get_elem_angular(par.element))
    bas = Basis(number, par, ihelp)
    alpha, coeff = bas.create_cgtos()

    j = torch.tensor(5)
    for i in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_gto(
                (torch.tensor(i), j),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
    i = torch.tensor(5)
    for j in range(5):
        with pytest.raises(IntegralTransformError):
            overlap_gto(
                (i, torch.tensor(j)),
                (alpha[0], alpha[1]),
                (coeff[0], coeff[1]),
                vec,
            )
