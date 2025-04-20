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
Parametrization: Module Types
=============================

Types for the differentiable representation of the extended tight-binding
parametrization using PyTorch.
"""
from __future__ import annotations

from torch import nn

from dxtb._src.typing import Any, Tensor

__all__ = ["NonNumericValue", "ParameterModule"]


class NonNumericValue(nn.Module):
    """
    Wraps a non-numeric value so it can be stored in an
    :class:`~torch.nn.Module`.

    This is used when the value is not to be trained (e.g. for strings).
    """

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value

    def forward(self) -> Any:
        """Return the stored non-numeric value."""
        return self.value

    def __getattr__(self, attr: str) -> Any:
        """
        Delegate attribute lookup to the underlying value.

        For example, if ``self.value`` is a string, then calling
        ``self.casefold()`` on a ``NonNumericValue`` instance will return
        ``self.value.casefold()``.
        """
        # __getattr__ is only called if attribute isn't found in usual places
        return getattr(self.value, attr)

    def __getitem__(self, key: Any) -> Any:
        """
        Delegate indexing to the underlying value.

        This allows you to directly use index operations on a
        ``NonNumericValue`` (for example, ``shell[-1]`` returns
        ``self.value[-1]``).
        """
        return self.value[key]

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.value})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


class ParameterModule(nn.Module):
    """
    Wraps a tensor as a trainable parameter (a :class:`~torch.nn.Parameter`)
    within a :class:`~torch.nn.Module`. Note that gradient tracking is
    **disabled** by default.

    This module ensures that each numeric leaf is stored as an
    :class:`~torch.nn.Module` so that it can be placed inside
    :class:`~torch.nn.ModuleDict` or :class:`~torch.nn.ModuleList`.
    """

    def __init__(self, value: Tensor) -> None:
        """
        Parameters
        ----------
        value : Tensor
            The tensor holding the numeric value.
        """
        super().__init__()
        self.param = nn.Parameter(value, requires_grad=False)

    def forward(self) -> Tensor:
        """Return the underlying parameter."""
        return self.param

    def __str__(self) -> str:  # pragma: no cover
        content = repr(self.param).replace("Parameter containing:\n", "")
        return f"{self.__class__.__name__}({content})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
