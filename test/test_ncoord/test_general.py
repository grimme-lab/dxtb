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
Test error handling in coordination number calculation.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._types import Any, CountingFunction, Protocol, Tensor
from dxtb.ncoord import (
    derf_count,
    dexp_count,
    erf_count,
    exp_count,
    get_coordination_number,
    get_coordination_number_gradient,
)


class CNFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        counting_function: CountingFunction = erf_count,
        rcov: Tensor | None = None,
        cutoff: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor: ...


class CNGradFunction(Protocol):
    """
    Type annotation for coordination number function.
    """

    def __call__(
        self,
        numbers: Tensor,
        positions: Tensor,
        dcounting_function: CountingFunction = derf_count,
        rcov: Tensor | None = None,
        cutoff: Tensor | None = None,
        **kwargs: Any,
    ) -> Tensor: ...


@pytest.mark.parametrize("function", [get_coordination_number])
@pytest.mark.parametrize("counting_function", [erf_count, exp_count])
def test_fail(function: CNFunction, counting_function: CountingFunction) -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        function(numbers, positions, counting_function, rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        function(numbers, positions, counting_function)


@pytest.mark.parametrize("function", [get_coordination_number_gradient])
@pytest.mark.parametrize("counting_function", [derf_count, dexp_count])
def test_grad_fail(
    function: CNGradFunction, counting_function: CountingFunction
) -> None:
    numbers = torch.tensor([1, 1])
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # rcov wrong shape
    with pytest.raises(ValueError):
        rcov = torch.tensor([1.0])
        function(numbers, positions, counting_function, rcov)

    # wrong numbers
    with pytest.raises(ValueError):
        numbers = torch.tensor([1])
        function(numbers, positions, counting_function)
