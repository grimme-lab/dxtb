"""
General tests for SCF setup.
"""
from __future__ import annotations

import torch

from dxtb.scf.iterator import SelfConsistentField


def test_properties() -> None:
    d = torch.tensor([1.0, 1.0, 1.0, 1.0])  # dummy
    scf = SelfConsistentField(d, d, d, d, d, d, d, d)  # type: ignore
    assert scf.shape == d.shape
    assert scf.device == d.device
    assert scf.dtype == d.dtype
