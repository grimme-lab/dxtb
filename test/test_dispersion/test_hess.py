# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Testing dispersion Hessian (autodiff).
"""

from __future__ import annotations

from math import sqrt

import pytest
import torch
from tad_mctc.autograd import jacrev
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb._src.components.classicals.dispersion import new_dispersion
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from ..utils import reshape_fortran
from .samples import samples

sample_list = ["LiH", "SiH4", "MB16_43_01", "PbH4-BiH3"]


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_single(dtype: torch.dtype, name: str) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps)

    sample = samples[name]
    numbers = sample["numbers"].to(DEVICE)
    positions = sample["positions"].to(**dd)
    ref = reshape_fortran(
        sample["hessian"].to(**dd),
        torch.Size(2 * (numbers.shape[-1], 3)),
    )
    numref = _numhess(numbers, positions)

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    def energy(pos: Tensor) -> Tensor:
        return disp.get_energy(pos, cache).sum()

    hess = jacrev(jacrev(energy))(pos)
    assert isinstance(hess, Tensor)

    pos.detach_()

    hess = hess.reshape_as(ref)
    assert ref.shape == numref.shape == hess.shape
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == numref.cpu()
    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name1", ["PbH4-BiH3"])
@pytest.mark.parametrize("name2", sample_list)
def skip_test_batch(dtype: torch.dtype, name1: str, name2) -> None:
    dd: DD = {"dtype": dtype, "device": DEVICE}
    tol = sqrt(torch.finfo(dtype).eps) * 100

    sample1, sample2 = samples[name1], samples[name2]
    numbers = pack(
        [
            sample1["numbers"].to(DEVICE),
            sample2["numbers"].to(DEVICE),
        ]
    )
    positions = pack(
        [
            sample1["positions"].to(**dd),
            sample2["positions"].to(**dd),
        ]
    )

    ref = pack(
        [
            reshape_fortran(
                sample1["hessian"].to(**dd),
                torch.Size(2 * (sample1["numbers"].to(DEVICE).shape[-1], 3)),
            ),
            reshape_fortran(
                sample2["hessian"].to(**dd),
                torch.Size(2 * (sample2["numbers"].shape[0], 3)),
            ),
        ]
    )

    # variable to be differentiated
    pos = positions.clone().requires_grad_(True)

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None

    cache = disp.get_cache(numbers)

    def energy(pos: Tensor) -> Tensor:
        return disp.get_energy(pos, cache).sum()

    hess = jacrev(jacrev(energy))(pos)
    assert isinstance(hess, Tensor)

    assert pytest.approx(ref.cpu(), abs=tol, rel=tol) == hess.detach().cpu()

    pos.detach_()


def _numhess(numbers: Tensor, positions: Tensor) -> Tensor:
    """Calculate numerical Hessian for reference."""
    dd = {"device": positions.device, "dtype": positions.dtype}

    disp = new_dispersion(numbers, par, **dd)
    assert disp is not None
    cache = disp.get_cache(numbers)

    hess = torch.zeros(*(*positions.shape, *positions.shape), **dd)
    step = 1.0e-4

    def _gradfcn(pos: Tensor) -> Tensor:
        pos.requires_grad_(True)
        energy = disp.get_energy(pos, cache)
        gradient = disp.get_gradient(energy, pos)
        pos.detach_()
        return gradient.detach()

    for i in range(numbers.shape[0]):
        for j in range(3):
            positions[i, j] += step
            gr = _gradfcn(positions)

            positions[i, j] -= 2 * step
            gl = _gradfcn(positions)

            positions[i, j] += step
            hess[:, :, i, j] = 0.5 * (gr - gl) / step

    return hess
