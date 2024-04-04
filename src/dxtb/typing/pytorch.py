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
Typing: PyTorch
===============

This module contains PyTorch-related type annotations.

Most importantly, the `TensorLike` base class is defined, which brings
tensor-like behavior (`.to` and `.type` methods) to classes.
"""
from tad_mctc.typing.pytorch import (
    DD,
    MockTensor,
    Molecule,
    Tensor,
    TensorLike,
    get_default_device,
    get_default_dtype,
)

__all__ = [
    "DD",
    "MockTensor",
    "Molecule",
    "Tensor",
    "TensorLike",
    "get_default_device",
    "get_default_dtype",
]
