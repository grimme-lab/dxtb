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
Typing: Compatibility
=====================

Since typing still significantly changes across different Python versions,
all the special cases are handled here.
"""
from __future__ import annotations

import sys

from tad_mctc.typing.compat import (
    Callable,
    CountingFunction,
    DampingFunction,
    Generator,
    PathLike,
    Self,
    Sequence,
    Size,
    Sliceable,
    TensorOrTensors,
    TypeGuard,
    override,
)
from tad_mctc.typing.pytorch import Tensor

__all__ = [
    "Callable",
    "CountingFunction",
    "DampingFunction",
    "Gather",
    "Generator",
    "PathLike",
    "Scatter",
    "ScatterOrGather",
    "Self",
    "Sequence",
    "Size",
    "Sliceable",
    "Slicer",
    "Tensor",
    "TensorOrTensors",
    "TypeGuard",
    "override",
]


# type aliases that do not require "from __future__ import annotations"
Gather = Callable[[Tensor, int, Tensor], Tensor]
Scatter = Callable[[Tensor, int, Tensor, str], Tensor]

if sys.version_info >= (3, 10):
    # "from __future__ import annotations" only affects type annotations
    # not type aliases, hence "|" is not allowed before Python 3.10
    ScatterOrGather = Gather | Scatter
    Slicer = list[slice] | tuple[slice] | tuple[type(...)]
elif sys.version_info >= (3, 9):
    # in Python 3.9, "from __future__ import annotations" works with type
    # aliases but requires using `Union` from typing
    from typing import Union

    ScatterOrGather = Union[Gather, Scatter]
    Slicer = Union[list[slice], tuple[slice], tuple[Ellipsis]]
elif sys.version_info >= (3, 8):
    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    from typing import List, Tuple, Union

    ScatterOrGather = Union[Gather, Scatter]
    Slicer = Union[List[slice], Tuple[slice], Tuple]
else:
    raise RuntimeError(
        f"'dxtb' requires at least Python 3.8 (Python {sys.version_info.major}."
        f"{sys.version_info.minor}.{sys.version_info.micro} found)."
    )
