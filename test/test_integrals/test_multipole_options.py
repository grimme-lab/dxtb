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
Batching, precision, chunking, screening and checkpointing of the PyTorch pair
builder.
"""

from __future__ import annotations

import torch
from tad_mctc.batch import pack

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.driver.pytorch.impls.os.compute_1d import compute_1d_os
from dxtb._src.integral.driver.pytorch.impls.pairs import (
    assemble_matrix,
    assemble_matrix_batch,
)
from dxtb._src.integral.driver.pytorch.impls.pipeline import (
    QUADRUPOLE_COMPONENTS,
)
from dxtb._src.typing import DD, Tensor

from ..conftest import DEVICE
from .samples import samples


def setup(numbers: Tensor, dd: DD) -> tuple[IndexHelper, list, list]:
    ihelp = IndexHelper.from_numbers(numbers, par)
    alphas, coeffs = Basis(numbers, par, ihelp, **dd).create_cgtos()
    return ihelp, alphas, coeffs


def two_molecules(dd: DD, gap: float) -> tuple[Tensor, Tensor]:
    """SiH4 and NH3, the latter shifted by ``gap`` along x."""
    shift = torch.tensor([gap, 0.0, 0.0], **dd)
    numbers = torch.cat([samples["SiH4"]["numbers"], samples["NH3"]["numbers"]])
    positions = torch.cat(
        [
            samples["SiH4"]["positions"].to(**dd),
            samples["NH3"]["positions"].to(**dd) + shift,
        ]
    )
    return numbers.to(DEVICE), positions


def test_batch() -> None:
    """
    A padded batch equals the single molecules, padding is exactly zero and
    gets no gradient. The planar molecule guards against removing padding
    based on coordinates (a zero z column).
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    nums = [samples[n]["numbers"].to(DEVICE) for n in ("SiH4", "LiH")]
    pos = [samples[n]["positions"].to(**dd) for n in ("SiH4", "LiH")]
    nums.append(torch.tensor([8, 1, 1], device=DEVICE))
    pos.append(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.43, 1.11, 0.0], [-1.43, 1.11, 0.0]], **dd
        )
    )

    numbers = pack(nums)
    positions = pack(pos).requires_grad_(True)
    out = assemble_matrix_batch(
        compute_1d_os, numbers, positions, par, QUADRUPOLE_COMPONENTS
    )

    for i, (n, p) in enumerate(zip(nums, pos)):
        ref = assemble_matrix_batch(
            compute_1d_os,
            n.unsqueeze(0),
            p.unsqueeze(0),
            par,
            QUADRUPOLE_COMPONENTS,
        )[0]
        nao = ref.shape[-1]
        assert torch.allclose(out[i, :, :nao, :nao], ref, atol=1e-14)
        assert (out[i, :, nao:, :] == 0).all()
        assert (out[i, :, :, nao:] == 0).all()

    (grad,) = torch.autograd.grad(out.sum(), positions)
    assert torch.isfinite(grad).all()
    assert (grad[numbers == 0] == 0).all()


def test_float32() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["SiH4"]["numbers"].to(DEVICE)
    positions = samples["SiH4"]["positions"].to(**dd)

    ref = assemble_matrix(
        compute_1d_os, *setup(numbers, dd), positions, QUADRUPOLE_COMPONENTS
    )

    dd = {"dtype": torch.float, "device": DEVICE}
    out = assemble_matrix(
        compute_1d_os,
        *setup(numbers, dd),
        positions.to(**dd),
        QUADRUPOLE_COMPONENTS,
    )

    assert out.dtype == torch.float
    assert (out.double() - ref).abs().max() <= 1e-5 * ref.abs().max()


def test_chunking_and_checkpointing() -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers, positions = two_molecules(dd, gap=5.0)
    ihelp, alphas, coeffs = setup(numbers, dd)

    results = []
    for kwargs in (
        {},
        {"chunk_size": 7},
        {"chunk_size": 7, "checkpoint": True},
    ):
        pos = positions.clone().requires_grad_(True)
        mat = assemble_matrix(
            compute_1d_os,
            ihelp,
            alphas,
            coeffs,
            pos,
            QUADRUPOLE_COMPONENTS,
            **kwargs,
        )
        (grad,) = torch.autograd.grad((mat**2).sum(), pos)
        results.append((mat.detach(), grad))

    for mat, grad in results[1:]:
        assert torch.allclose(mat, results[0][0], atol=1e-14, rtol=0.0)
        assert torch.allclose(grad, results[0][1], atol=1e-11, rtol=0.0)


def test_screening() -> None:
    """
    Screening drops the blocks of distant shell pairs exactly, keeps the
    others unchanged, and does not detach positions or basis parameters.
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers, positions = two_molecules(dd, gap=40.0)
    ihelp, alphas0, coeffs0 = setup(numbers, dd)

    results = []
    for threshold in (None, 1e-14):
        pos = positions.clone().requires_grad_(True)
        alphas = [a.detach().clone().requires_grad_(True) for a in alphas0]
        coeffs = [c.detach().clone().requires_grad_(True) for c in coeffs0]
        mat = assemble_matrix(
            compute_1d_os,
            ihelp,
            alphas,
            coeffs,
            pos,
            QUADRUPOLE_COMPONENTS,
            screening_threshold=threshold,
        )
        grads = torch.autograd.grad((mat**2).sum(), [pos, *alphas, *coeffs])
        results.append((mat.detach(), grads))

    (ref, ref_grads), (scr, scr_grads) = results

    nao = 17  # SiH4
    assert (scr[:, :nao, nao:] == 0).all()
    assert (scr[:, :nao, :nao] == ref[:, :nao, :nao]).all()
    assert (scr - ref).abs().max() < 1e-12

    for g_ref, g_scr in zip(ref_grads, scr_grads):
        assert torch.allclose(g_scr, g_ref, atol=1e-10, rtol=1e-10)
