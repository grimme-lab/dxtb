"""
Testing the charges module
==========================

This module tests the EEQ charge model including:
 - single molecule
 - batched
 - ghost atoms
 - autograd via `gradcheck`

Note that `torch.linalg.solve` gives slightly different results (around 1e-5
to 1e-6) across different PyTorch versions (1.11.0 vs 1.13.0) for single
precision. For double precision, however the results are identical.
"""
from __future__ import annotations

import pytest
import torch

from dxtb import charges
from dxtb._types import Tensor

from .samples import samples


@pytest.mark.grad
def test_gradcheck(dtype: torch.dtype = torch.double):
    sample = samples["NH3"]
    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    total_charge = sample["total_charge"].type(dtype)
    cn = torch.tensor(
        [3.0, 1.0, 1.0, 1.0],
        dtype=dtype,
    )
    eeq = charges.ChargeModel.param2019().type(dtype)

    positions.requires_grad_(True)
    total_charge.requires_grad_(True)

    def func(positions: Tensor, total_charge: Tensor):
        return torch.sum(
            charges.solve(numbers, positions, total_charge, eeq, cn)[0], -1
        )

    # pylint: disable=import-outside-toplevel
    from torch.autograd.gradcheck import gradcheck

    assert gradcheck(func, (positions, total_charge))

    positions.detach_()
    total_charge.detach()
