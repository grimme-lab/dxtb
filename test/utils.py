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
Collection of utility functions for testing.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tad_mctc.data import pse

from dxtb._src.param.element import Element
from dxtb._src.typing import Any, Size, Tensor

coordfile = Path(
    Path(__file__).parent, "test_singlepoint/mols/H2/coord"
).resolve()
"""Path to coord file of H2."""

coordfile_lih = Path(
    Path(__file__).parent, "test_singlepoint/mols/LiH/coord"
).resolve()
"""Path to coord file of LiH."""


def load_from_npz(
    npzfile: Any,
    name: str,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> Tensor:
    """Get torch tensor from npz file

    Parameters
    ----------
    npzfile : Any
        Loaded npz file.
    name : str
        Name of the tensor in the npz file.
    dtype : torch.dtype
        Data type of the tensor.
    device : torch.device | None
        Device of the tensor. Defaults to ``None``.

    Returns
    -------
    Tensor
        Tensor from the npz file.
    """
    name = name.replace("-", "").replace("+", "").lower()
    return torch.from_numpy(npzfile[name]).to(device=device, dtype=dtype)


def load_from_tblite_grad(
    file: Path,
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> dict[str, Tensor]:
    """
    Load a TBLITE gradient file and convert it to a dictionary of tensors.

    Parameters
    ----------
    file : Path
        Path to the TBLITE gradient file.
    dtype : torch.dtype
        Data type of the tensors.
    device : torch.device | None, optional
        Device to store the tensors. If ``None``, the default device is used.

    Returns
    -------
    dict[str, Tensor]
        Dictionary mapping keys to tensors.
    """
    tensor_dict = {}
    current_key = None
    current_shape = ""
    current_data = []

    with open(file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Identify header lines by checking for ":real:"
            if ":real:" in line:
                # If processing a previous field, convert collected data into a tensor.
                if current_key is not None:
                    flat_vals = [float(x) for x in current_data]
                    tensor = torch.tensor(flat_vals, device=device, dtype=dtype)
                    if current_shape:
                        # Convert shape string (e.g. "3,39") to a tuple of ints.
                        shape = tuple(int(s) for s in current_shape.split(","))
                        # Invert the shape if there is more than one dimension.
                        if len(shape) > 1:
                            shape = tuple(reversed(shape))
                        tensor = tensor.view(shape)
                    tensor_dict[current_key] = tensor

                # Parse header using simple splitting.
                # Format is: key :real:dim:shape
                parts = line.split()
                current_key = parts[0]
                header_parts = line.split(":")
                # header_parts[3] is the shape information (if any)
                current_shape = (
                    header_parts[3].strip() if len(header_parts) > 3 else ""
                )
                current_data = []
            else:
                # Append data values from non-header lines.
                current_data.extend(line.split())

        # Process the final block after file ends.
        if current_key is not None:
            flat_vals = [float(x) for x in current_data]
            tensor = torch.tensor(flat_vals, dtype=torch.float32)
            if current_shape:
                shape = tuple(int(s) for s in current_shape.split(","))
                if len(shape) > 1:
                    shape = tuple(reversed(shape))
                tensor = tensor.view(shape)
            tensor_dict[current_key] = tensor

    return tensor_dict


def nth_derivative(f: Tensor, x: Tensor, n: int = 1) -> Tensor:
    """
    Calculate the *n*th derivative of a tensor.

    Parameters
    ----------
    f : Tensor
        Input tensor of which the gradient should be calculated.
    x : Tensor
        Dependent variable (must have `requires_grad`)
    n : int, optional
        Order of the derivative. Defaults to 1.

    Returns
    -------
    Tensor
        The *n*th order derivative of `f` w.r.t. `x`.

    Raises
    ------
    ValueError
        Order of derivative is smaller than 1 or not an integer.
    """
    if n < 1 or not isinstance(n, int):
        raise ValueError("Order of derivative must be an integer and larger 1.")

    create_graph = False if n == 1 else True

    grads = None
    for _ in range(n):
        (grads,) = torch.autograd.grad(f, x, create_graph=create_graph)
        f = grads.sum()

    assert grads is not None
    return grads


def reshape_fortran(x: Tensor, shape: Size) -> Tensor:
    """
    Implements Fortran's `reshape` function (column-major).

    Parameters
    ----------
    x : Tensor
        Input tensor
    shape : Size
        Output size to which `x` is reshaped.

    Returns
    -------
    Tensor
        Reshaped tensor of size `shape`.
    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def get_elem_param(
    unique: Tensor,
    par_element: dict[str, Element],
    key: str,
    pad_val: int = -1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Obtain a element-wise parametrized quantity for selected atomic numbers.

    Parameters
    ----------
    unique : Tensor
        Unique atomic numbers in the system (shape: ``(nunique,)``).
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).
    pad_val : int, optional
        Value to pad the tensor with. Default is `-1`.
    device : torch.device | None
        Device to store the tensor. If ``None`` (default), the default
        device is used.
    dtype : torch.dtype | None
        Data type of the tensor. If ``None`` (default), the data type
        is inferred.

    Returns
    -------
    Tensor
        Parametrization of selected elements.

    Raises
    ------
    ValueError
        If the type of the value of `key` is neither `float` nor `int`.
    """
    l = []

    for number in unique:
        el = pse.Z2S.get(int(number.item()), "X")
        if el in par_element:
            p = par_element[el]
            if key not in p.model_fields:
                raise KeyError(
                    f"The key '{key}' is not in the element parameterization"
                )
            vals = getattr(p, key)

            # convert to list so that we can use the same function
            # for atom-resolved parameters too
            if isinstance(vals, float):
                vals = [vals]

            if not all(isinstance(x, (int, float)) for x in vals):
                raise ValueError(
                    f"The key '{key}' contains the non-numeric values '{vals}'."
                )

        else:
            vals = [pad_val]

        for val in vals:
            l.append(val)

    return torch.tensor(l, device=device, dtype=dtype)
