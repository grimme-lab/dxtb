"""
General tests for SCF setup.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.integral import IntegralMatrices
from dxtb.scf.iterator import SelfConsistentField


def test_properties() -> None:
    d = torch.randn((3, 3))  # dummy

    ints = IntegralMatrices()
    with pytest.raises(RuntimeError):
        SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.hcore = torch.randn((3, 3))
    with pytest.raises(RuntimeError):
        SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore

    ints.overlap = torch.randn((3, 3))
    scf = SelfConsistentField(d, d, d, d, d, d, integrals=ints)  # type: ignore
    assert scf.shape == d.shape
    assert scf.device == d.device
    assert scf.dtype == d.dtype
