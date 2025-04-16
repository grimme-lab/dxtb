# This file is part of dxtb.
#
# SPDX-License-Identifier: Apache-2.0
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
Test the differentiable parameter model.
"""

import pytest
import torch
from torch import nn

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule
from dxtb._src.param.module import ParameterModule


def test_ParamModule_creation() -> None:
    """Test that a ParamModule can be created from a valid parameter model."""
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)

    tree = diff_param.forward()
    assert isinstance(tree, nn.ModuleDict)


def test_get_valid() -> None:
    """Test that a valid key path returns the underlying tensor."""
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)
    gexp = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should be returned as a tensor."
    assert isinstance(gexp, torch.Tensor), msg

    # By default, parameters are created with gradients disabled.
    msg = "gexp should have requires_grad == False by default."
    assert gexp.requires_grad is False, msg


def test_get_invalid_key() -> None:
    """Test that using a non-existent key produces a KeyError."""
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)

    with pytest.raises(KeyError):
        diff_param.get("nonexistent")


def test_get_wrong_type() -> None:
    """
    Non-string key for ModuleDict (or non-int for ModuleList) raises TypeError.
    """
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)

    with pytest.raises(TypeError):
        diff_param.get(123)


def test_set_differentiable_leaf() -> None:
    """
    Test that setting a leaf node (e.g. charge->effective->gexp) to be
    differentiable updates the tensor.
    """
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)

    # Retrieve gexp and verify default gradient flag.
    gexp = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should initially be non-differentiable."
    assert gexp.requires_grad is False, msg

    # Set the leaf to be differentiable.
    diff_param.set_differentiable("charge", "effective", "gexp")
    gexp_after = diff_param.get("charge", "effective", "gexp")

    msg = "gexp should be set to differentiable."
    assert gexp_after.requires_grad is True, msg


def test_set_differentiable_branch_ignore_non_numeric() -> None:
    """
    Test that setting an entire branch (e.g. 'charge') to be differentiable with ignore_non_numeric=True
    recursively updates all numeric leaves, ignoring static leaves.
    """
    par = GFN1_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)

    # Set entire 'charge' branch to differentiable while ignoring static fields.
    diff_param.set_differentiable("charge", ignore_non_numeric=True)

    def check_diff(module: nn.Module) -> None:
        if isinstance(module, ParameterModule):
            msg = "All numeric parameters should be differentiable."
            assert module.param.requires_grad is True, msg
        elif isinstance(module, (nn.ModuleDict, nn.ModuleList)):
            for child in module.children():
                check_diff(child)
        # StaticValue is ignored.

    branch = diff_param.get("charge", unwrapped=False)
    check_diff(branch)


def test_set_differentiable_failure() -> None:
    """
    Test that attempting to set a non-numeric (static) field to be
    differentiable raises a TypeError when ignore_non_numeric is False.
    For example, 'meta'->'name' is expected to be a static string.
    """
    par = GFN2_XTB.model_copy(deep=True)
    diff_param = ParamModule(par, dtype=torch.float64)
    with pytest.raises(TypeError):
        diff_param.set_differentiable("meta", "name", ignore_non_numeric=False)
