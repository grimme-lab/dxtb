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
Parametrization: Differentiable Parametrization
===============================================

This module provides a differentiable representation of the
extended tight-binding parametrization using PyTorch.

The :class:`.ParamModule` class automatically converts a Pydantic
model into a hierarchical :class:`~torch.nn.Module` tree.
"""
from __future__ import annotations

import torch
from torch import nn

from dxtb._src.constants.defaults import DEFAULT_BASIS_INT
from dxtb._src.typing import Any, Self, get_default_dtype
from dxtb._src.utils import is_float, is_float_list, is_int_list, is_integer

from ..base import Param
from .types import NonNumericValue, ParameterModule
from .utils import ParamElementsPairsMixin

__all__ = ["NonNumericValue", "ParamModule", "ParameterModule"]


class ParamModule(nn.Module, ParamElementsPairsMixin):
    """
    Automatically converts a Pydantic model into a hierarchical
    :class:`~torch.nn.Module` tree. All numeric values (or lists of numbers)
    are wrapped in :class:`ParameterModule` instances (with
    :attr:`requires_grad` set to False by default) so that they are registered
    as parameters. Nonnumeric values are wrapped in :class:`NonNumericValue`.
    """

    def __init__(
        self,
        par: Param,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Parameters
        ----------
        par : Param
            The validated Pydantic model.
        device : torch.device | None, optional
            Device of the tensors. Defaults to ``None``.
        dtype : torch.dtype | None, optional
            Data type of the tensors. If ``None``, the default data type from
            ``get_default_dtype`` is used. Defaults to ``None``.
        """
        super().__init__()

        # Recursively convert the dictionary into a parameter tree.
        self.parameter_tree = _auto_convert(
            par.clean_model_dump(), device, dtype
        )

        self.device = device
        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.dd = {"device": self.device, "dtype": self.dtype}

    def _apply(self, fn, recurse: bool = True) -> Self:
        # Call the parent _apply to update the module's parameters and buffers.
        # https://stackoverflow.com/questions/54706146/moving-member-tensors-with-module-to-in-pytorch
        super()._apply(fn, recurse=recurse)
        self.device = fn(self.device)
        self.dtype = fn(self.dtype)
        self.dd = {"device": self.device, "dtype": self.dtype}

        return self

    def forward(self) -> nn.Module:
        """
        Returns the internal parameter tree. This tree contains all numeric
        values wrapped in  :class:`ParameterModule` and nonnumeric values
        wrapped in :class:`NonNumericValue`.
        """
        return self.parameter_tree

    # Pretty-printing

    def __str__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}({self.parameter_tree})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)


def _auto_convert(
    value: Any,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """
    Recursively converts an input value into a :class:`~torch.nn.Module` tree.

    * If the input is a dictionary, returns a :class:`~torch.nn.ModuleDict`
      with values recursively converted.
    * If the input is a list of numbers, converts it into a tensor wrapped in
      a :class:`ParameterModule`.
    * For a nonnumeric list, each element is converted; if all are modules,
      they are put into a :class:`~torch.nn.ModuleList`.
      Otherwise the list is wrapped in a :class:`NonNumericValue`.
    * If the input is a numeric value, converts it into a tensor wrapped in a
      :class:`ParameterModule`.
    * All nonnumeric values are wrapped in a :class:`NonNumericValue`.

    Parameters
    ----------
    value : Any
        The value to convert.
    device : torch.device | None, optional
        Device of the tensors. If ``None``, the device of `freqs` is used.
        Defaults to ``None``.
    dtype : torch.dtype | None, optional
        Data type of the tensors. If ``None``, the data type of `freqs` is
        used. Defaults to ``None``.

    Returns
    -------
    nn.Module
        The converted value as a :class:`~torch.nn.Module`.
    """
    # If value is a dictionary, convert each key recursively.
    if isinstance(value, dict):
        out = nn.ModuleDict()
        for key, v in value.items():
            out[key] = _auto_convert(v, device, dtype)
        return out

    # If value is a list...
    if isinstance(value, list):
        # If it's a list of numbers, convert it directly.
        _is_int = is_int_list(value)
        _is_float = is_float_list(value)
        if value and (_is_int or _is_float):
            _dtype = dtype if _is_float else DEFAULT_BASIS_INT
            tensor_value = torch.tensor(value, device=device, dtype=_dtype)
            return ParameterModule(tensor_value)

        # Otherwise, process each item recursively.
        converted: list[nn.Module] = []
        all_modules = True
        for item in value:
            conv_item = _auto_convert(item, device, dtype)
            if not isinstance(conv_item, nn.Module):
                all_modules = False
            converted.append(conv_item)

        if all_modules:
            return nn.ModuleList(converted)

        return NonNumericValue(converted)

    # If value is a numeric leaf, convert it to a ParameterModule.
    if is_float(value):
        tensor_value = torch.tensor(value, device=device, dtype=dtype)
        return ParameterModule(tensor_value)

    if is_integer(value):
        tensor_value = torch.tensor(value, device=device, dtype=torch.int8)
        return ParameterModule(tensor_value)

    # For any other type, wrap in NonNumericValue.
    return NonNumericValue(value)
