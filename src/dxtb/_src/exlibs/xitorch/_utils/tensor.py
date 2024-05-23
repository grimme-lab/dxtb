# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
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
This file contains functions for some linear algebra and basic operations of
torch.tensor.
"""

import torch

__all__ = [
    "tallqr",
    "to_fortran_order",
    "convert_none_grads_to_zeros",
    "create_random_square_matrix",
    "create_random_ortho_matrix",
]


def tallqr(V, MV=None):
    # faster QR for tall and skinny matrix
    # V: (*BV, na, nguess)
    # MV: (*BM, na, nguess) where M is the basis to make Q M-orthogonal
    # if MV is None, then MV=V
    if MV is None:
        MV = V
    VTV = torch.matmul(V.transpose(-2, -1), MV)  # (*BMV, nguess, nguess)
    R = (
        torch.linalg.cholesky(VTV.transpose(-2, -1).conj()).transpose(-2, -1).conj()
    )  # (*BMV, nguess, nguess)
    Rinv = torch.inverse(R)  # (*BMV, nguess, nguess)
    Q = torch.matmul(V, Rinv)
    return Q, R


def to_fortran_order(V):
    # V: (...,nrow,ncol)
    # outV: (...,nrow,ncol)

    # check if it is in C-contiguous
    if V.is_contiguous():
        # return V.set_(V.storage(), V.storage_offset(), V.size(), tuple(reversed(V.stride())))
        return V.transpose(-2, -1).contiguous().transpose(-2, -1)
    elif V.transpose(-2, -1).is_contiguous():
        return V
    else:
        raise RuntimeError("Only the last two dimensions can be made Fortran order.")


def convert_none_grads_to_zeros(grads, inputs):
    is_tuple = isinstance(grads, tuple)
    if is_tuple:
        grads = list(grads)
    for i in range(len(grads)):
        g = grads[i]
        if g is None:
            grads[i] = torch.zeros_like(inputs[i])
    if is_tuple:
        grads = tuple(grads)
    return grads


def create_random_square_matrix(
    n, is_hermitian=False, min_eival=1.0, max_eival=1.0, minabs_eival=0.0, seed=-1
):
    dtype = torch.float64
    eivals = torch.linspace(min_eival, max_eival, n, dtype=dtype)

    # clip absolute eivals to minabs_eival
    idx = eivals.abs() < minabs_eival
    eivals[idx] = torch.sign(eivals[idx]) * minabs_eival

    eivals = torch.diag_embed(eivals)
    if seed > 0:
        torch.manual_seed(seed)
    if is_hermitian:
        eivecs = create_random_ortho_matrix(n, seed=seed)
        mat = torch.matmul(torch.matmul(eivecs.transpose(-2, -1), eivals), eivecs)
        mat = (mat + mat.transpose(-2, -1)) * 0.5
        return mat
    else:
        a = torch.randn((n, n), dtype=dtype)
        a = a / a.norm(dim=-2, keepdim=True)
        return torch.matmul(torch.matmul(a.inverse(), eivals), a)


def create_random_ortho_matrix(n, seed=-1):
    dtype = torch.float64
    if seed > 0:
        torch.manual_seed(seed)
    a = torch.randn((n, n), dtype=dtype)
    q, r = torch.linalg.qr(a)
    return q
