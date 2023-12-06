"""
Test Calculator setup.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.param import GFN1_XTB as par
from dxtb.timing import timer
from dxtb.utils import DtypeError
from dxtb.xtb import Calculator


def test_fail() -> None:
    numbers = torch.tensor([6, 1, 1, 1, 1], dtype=torch.double)

    with pytest.raises(DtypeError):
        Calculator(numbers, par)

    # because of the exception, the timer for the setup is never stopped
    timer.reset()
