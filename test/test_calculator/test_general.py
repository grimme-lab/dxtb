"""
Test Calculator setup.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.xtb import Calculator


def test_fail() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1])
    calc = Calculator(numbers, par)

    with pytest.raises(KeyError):
        calc.set_option("something", 1)

    with pytest.raises(KeyError):
        calc.set_tol("something", 1)
