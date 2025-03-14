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
Test the mixer.
"""

from __future__ import annotations

import pytest
import torch

from dxtb import OutputHandler
from dxtb._src.scf.mixer import Anderson, Mixer, Simple
from dxtb._src.typing import Tensor

from ..conftest import DEVICE


def func(x: Tensor) -> Tensor:
    """
    Non-linear convergence test function.

    This function, although simple, is commonly used to test the validity of
    mixing algorithms. A initial ``x`` vector of ones should be used, & should
    eventually converge to zero. However, it should be noted that directly
    calling this function cyclically without the aid of a mixing algorithm
    will result in divergent behaviour.
    """
    d = torch.tensor([3.0, 2.0, 1.5, 1.0, 0.5], dtype=x.dtype, device=x.device)
    c = 0.01
    return x + (-d * x - c * x**3)


def general(mixer: Mixer, device: torch.device | None) -> None:
    """Tests some of the mixer's general operational functionality.

    This tests the basic operational functionality of a mixer. It should be
    noted that this is not a comprehensive test. This is intended to catch
    basic operating errors. Once this function is complete it will call the
    ``mixer.reset()`` function. It is important to check that the mixer has
    actually be reset. Each check must be done individually for each mixer
    as it is a highly mixer specific task.

    Tests:
        1. Tolerance condition's warning and clip subroutines.
        2. Mixing can be performed & returns the correct shape.
        3. Step number is incremented.
        4. Delta values are returned correctly.
        5. Convergence property returns expected result.
        6. Cull operation runs as anticipated.
        7. Reset function works as expected.

    Args:
        mixer: The mixer to test
        device: the device to run on

    """
    name = mixer.__class__.__name__
    nel = 5
    a = torch.ones(nel, nel, nel, device=device)
    a_copy = a.clone()
    mixer._batch_mode = True

    # Check 1 removed, only 2
    for _ in range(10):
        a = mixer.iter(func(a), a)
        chk_2 = a.shape == a_copy.shape
        assert chk_2, f"{name} Input & output shapes do not match"

    # Check 3
    chk_3 = mixer.iter_step != 0
    assert chk_3, f"{name}.iter_step was not incremented"

    # Check 4
    b = a
    c = func(a)
    a = mixer.iter(c, a)
    chk_4a = mixer.delta.shape == a_copy.shape
    chk_4b = torch.allclose(c - b, mixer.delta)
    assert chk_4a, f"{name}.delta has an incorrect shape"
    assert chk_4b, f"{name}.delta values are not correct"

    # Check 5
    converged = mixer.converged
    chk_5a = converged.shape == a_copy.shape[0:1]
    chk_5b = converged.any() == False  # Should not have converged yet
    a = mixer.iter(a, a)
    chk_5c = mixer.converged.all() == True
    assert chk_5a, f"{name}.converged has an incorrect shape"
    assert chk_5b, f"{name}.converged should all be False"
    assert chk_5c, f"{name}.converged should all be True"

    # Check 6
    cull_list = torch.tensor([True, False, True, False, True], device=device)
    mixer.cull(cull_list, mpdim=5)
    # Next mixer call should crash if cull was not implemented correctly.
    a_culled = a[~cull_list]
    a = mixer.iter(func(a_culled), a_culled)

    chk_6 = a.shape == a_copy[~cull_list].shape
    assert chk_6, f"{name} returned an unexpected shape after cull operation"

    # Check 7
    # This only performs the reset operation to catch any fatal errors.
    # Additional checks must be performed in the mixer specific function.
    mixer.reset()
    assert mixer.iter_step == 0, f"{name}.iter_step was not reset"
    assert mixer._delta is None, f"{name}._delta was not reset"


def test_simple() -> None:
    opts = {"damp": 0.05}
    general(Simple(opts, batch_mode=1), DEVICE)


def test_anderson() -> None:
    # tbmalt default settings
    opts = {
        "damp": 0.05,
        "damp_init": 0.01,
        "generations": 4,
        "diagonal_offset": 0.01,
    }
    general(Anderson(opts, batch_mode=1), DEVICE)


def test_fail_dim_batch() -> None:
    mixer = Simple(batch_mode=1)
    x_new = torch.ones(3, 2, 2)
    x_old = torch.ones(1, 2, 2)

    with pytest.raises(RuntimeError):
        mixer.iter(x_new, x_old)


def test_fail_dim_nonbatch() -> None:
    mixer = Simple(batch_mode=1)
    x_new = torch.ones(3, 4, 2)
    x_old = torch.ones(3, 2, 5)

    with pytest.raises(RuntimeError):
        mixer.iter(x_new, x_old)


def test_fail_delta() -> None:
    mixer = Simple()

    with pytest.raises(RuntimeError):
        _ = mixer.delta

    with pytest.raises(RuntimeError):
        _ = mixer.delta_norm

    with pytest.raises(RuntimeError):
        _ = mixer.converged


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tolerances_1(dtype: torch.dtype) -> None:
    opts = {"x_tol": 1e-30}
    mixer = Simple(opts)

    x = torch.ones(5, dtype=dtype)
    mixer.iter(x, x)

    assert mixer.options["x_tol"] == torch.finfo(dtype).resolution * 50

    assert len(OutputHandler.warnings) == 1

    warn_msg, warn_type = OutputHandler.warnings[0]
    assert "x_tol=1e-30" in warn_msg
    assert warn_type is UserWarning

    OutputHandler.clear_warnings()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tolerances_2(dtype: torch.dtype) -> None:
    opts = {"x_tol_max": 1e-30}
    mixer = Simple(opts)

    x = torch.ones(5, dtype=dtype)
    mixer.iter(x, x)

    assert mixer.options["x_tol_max"] == torch.finfo(dtype).resolution * 50

    assert len(OutputHandler.warnings) == 1

    warn_msg, warn_type = OutputHandler.warnings[0]
    assert "x_tol_max=1e-30" in warn_msg
    assert warn_type is UserWarning

    OutputHandler.clear_warnings()
