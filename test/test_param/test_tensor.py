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
Test the tensor serialization.
"""

from __future__ import annotations

import torch
from pydantic import BaseModel

from dxtb._src.param.tensor import TensorPydantic

from ..conftest import DEVICE


class Model(BaseModel):
    """Dummy model to test serialization."""

    tensor: TensorPydantic


def test_array_serialization() -> None:
    """Test serialization of a 2D tensor."""
    t = torch.tensor([[1, 2], [3, 4]], device=DEVICE)
    model = Model(tensor=t)  # type: ignore

    expected = {"tensor": t.tolist()}
    assert model.model_dump() == expected


def test_scalar_serialization() -> None:
    """Test serialization of a scalar tensor."""
    scalar = torch.tensor(42, device=DEVICE)
    model = Model(tensor=scalar)  # type: ignore

    expected = {"tensor": int(scalar.item())}
    assert model.model_dump() == expected


def test_grad_tensor_serialization() -> None:
    """Test serialization of a tensor with gradients."""
    t = torch.tensor([[1.0, 4.0]], device=DEVICE, requires_grad=True)
    model = Model(tensor=t)  # type: ignore

    expected = {"tensor": t.tolist()}
    assert model.model_dump() == expected
