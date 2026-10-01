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
Overlap, dipole and quadrupole matrices of the PyTorch pair builder
(``assemble_matrix``): libcint reference, physical relations and derivatives.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import jacfwd, jacrev

from dxtb import GFN1_XTB as par
from dxtb import IndexHelper
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.pytorch.impls.algorithms import (
    ALGORITHMS,
    get_kernel,
)
from dxtb._src.integral.driver.pytorch.impls.pairs import assemble_matrix
from dxtb._src.integral.driver.pytorch.impls.pipeline import (
    DIPOLE_COMPONENTS,
    QUADRUPOLE_COMPONENTS,
)
from dxtb._src.typing import DD, Tensor
from dxtb.integrals.wrappers import dipint, overlap, quadint
from dxtb.labels import INTDRIVER_LIBCINT

from ..conftest import DEVICE
from .samples import samples

COMPONENTS = {
    "overlap": None,
    "dipole": DIPOLE_COMPONENTS,
    "quadrupole": QUADRUPOLE_COMPONENTS,
}

# libcint order -> pytorch order of the components within a shell
PERM_BY_L = {0: [0], 1: [1, 2, 0], 2: [0, 1, 2, 3, 4]}


class Matrices:
    """Integral matrices of one molecule with one algorithm."""

    def __init__(self, numbers: Tensor, algorithm: str, dd: DD) -> None:
        self.ihelp = IndexHelper.from_numbers(numbers, par)
        bas = Basis(numbers, par, self.ihelp, **dd)
        self.alphas, self.coeffs = bas.create_cgtos()
        self.kernel = get_kernel(algorithm)

    def __call__(
        self, positions: Tensor, kind: str, origin: Tensor | None = None
    ) -> Tensor:
        return assemble_matrix(
            self.kernel,
            self.ihelp,
            self.alphas,
            self.coeffs,
            positions,
            COMPONENTS[kind],
            origin,
        )


def permutation(ihelp: IndexHelper, dd: DD) -> Tensor:
    angular = ihelp.angular.tolist()
    orbs = ihelp.orbitals_per_shell.tolist()

    p = torch.zeros((sum(orbs), sum(orbs)), **dd)
    off = 0
    for l, nao in zip(angular, orbs):
        for k, pk in enumerate(PERM_BY_L[l]):
            p[off + k, off + pk] = 1.0
        off += nao
    return p


@pytest.mark.skipif(not has_libcint, reason="libcint not available")
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("name", ["LiH", "SiH4", "C6H5I-CH3SH"])
def test_matches_libcint(name: str, algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples[name]["numbers"].to(DEVICE)
    positions = samples[name]["positions"].to(**dd)

    mats = Matrices(numbers, algorithm, dd)
    p = permutation(mats.ihelp, dd)

    for kind, ref_fn in (
        ("overlap", overlap),
        ("dipole", dipint),
        ("quadrupole", quadint),
    ):
        ref = ref_fn(numbers, positions, par, driver=INTDRIVER_LIBCINT)
        ref = p @ ref.to(DEVICE) @ p.mT
        if kind == "overlap":
            ref = ref.unsqueeze(0)

        atol = 1e-12 * max(1.0, float(ref.abs().max()))
        assert torch.allclose(mats(positions, kind), ref, atol=atol, rtol=0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_symmetry_translation_and_origin_shift(algorithm: str) -> None:
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["SiH4"]["numbers"].to(DEVICE)
    positions = samples["SiH4"]["positions"].to(**dd)
    mats = Matrices(numbers, algorithm, dd)

    s = mats(positions, "overlap")
    d = mats(positions, "dipole")
    q = mats(positions, "quadrupole")

    for m in (s, d, q):
        assert torch.allclose(m, m.mT, atol=1e-13, rtol=0.0)
    assert torch.allclose(
        torch.diagonal(s[0]), torch.ones(s.shape[-1], **dd), atol=1e-13
    )

    # translation: the overlap is invariant, the dipole picks up t * S
    t = torch.tensor([0.4, -1.3, 0.9], **dd)
    assert torch.allclose(mats(positions + t, "overlap"), s, atol=1e-13)
    d_t = mats(positions + t, "dipole")
    assert torch.allclose(d_t, d + t[:, None, None] * s, atol=1e-12)

    # origin C: <(r - C)_a> = D_a - C_a S and
    # <(r - C)_a (r - C)_b> = Q_ab - C_a D_b - C_b D_a + C_a C_b S
    c = torch.tensor([0.3, -0.7, 1.1], **dd)
    d_c = mats(positions, "dipole", c)
    q_c = mats(positions, "quadrupole", c)
    for a in range(3):
        assert torch.allclose(d_c[a], d[a] - c[a] * s[0], atol=1e-12)
        for b in range(3):
            ref = q[3 * a + b] - c[a] * d[b] - c[b] * d[a] + c[a] * c[b] * s[0]
            assert torch.allclose(q_c[3 * a + b], ref, atol=1e-12)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_single_atom(algorithm: str) -> None:
    """Parity selection rules for coincident centers, and finite gradients."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = torch.tensor([14], device=DEVICE)  # s, p, d
    positions = torch.zeros((1, 3), **dd, requires_grad=True)
    mats = Matrices(numbers, algorithm, dd)

    ls = torch.repeat_interleave(mats.ihelp.angular, 2 * mats.ihelp.angular + 1)
    odd = (ls[:, None] - ls[None, :]) % 2 == 1

    d = mats(positions, "dipole")
    q = mats(positions, "quadrupole")
    assert (d[:, ~odd].abs() < 1e-13).all() and d.abs().max() > 0.1
    assert (q[:, odd].abs() < 1e-13).all() and q.abs().max() > 0.1

    (grad,) = torch.autograd.grad(q.sum(), positions)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_rotation_covariance(algorithm: str) -> None:
    """
    A rotation mixes the components within a shell, so the basis-independent
    traces :math:`t_k = \\mathrm{Tr}(S^{-1} D_k)` and
    :math:`T_{kl} = \\mathrm{Tr}(S^{-1} Q_{kl})` are compared instead, which
    transform as a vector and a rank-2 tensor.
    """
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["SiH4"]["numbers"].to(DEVICE)
    positions = samples["SiH4"]["positions"].to(**dd)
    positions = positions + torch.tensor([0.4, -0.3, 0.7], **dd)
    mats = Matrices(numbers, algorithm, dd)

    gen = torch.Generator(device="cpu").manual_seed(3)
    rand = torch.randn(3, 3, generator=gen, dtype=torch.double, device="cpu")
    rot = torch.linalg.qr(rand)[0].to(**dd)

    def traces(pos: Tensor) -> tuple[Tensor, Tensor]:
        s_inv = torch.linalg.inv(mats(pos, "overlap")[0])
        t = torch.einsum("ij,kji->k", s_inv, mats(pos, "dipole"))
        tt = torch.einsum("ij,kji->k", s_inv, mats(pos, "quadrupole"))
        return t, tt.reshape(3, 3)

    t0, tt0 = traces(positions)
    t1, tt1 = traces(positions @ rot.T)

    assert t0.abs().max() > 1e-3
    assert torch.allclose(t1, rot @ t0, atol=1e-9, rtol=0.0)
    assert torch.allclose(tt1, rot @ tt0 @ rot.T, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_func_hessian(algorithm: str) -> None:
    """``torch.func`` Hessians (reverse-reverse and forward-reverse)."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2O"]["numbers"].to(DEVICE)
    positions = samples["H2O"]["positions"].to(**dd)
    mats = Matrices(numbers, algorithm, dd)

    gen = torch.Generator(device="cpu").manual_seed(0)
    nao = int(mats.ihelp.orbitals_per_shell.sum())
    weights = torch.randn(9, nao, nao, generator=gen, device="cpu").to(**dd)

    def scalar(pos: Tensor) -> Tensor:
        return (mats(pos, "quadrupole") * weights).sum()

    ref = torch.autograd.functional.hessian(scalar, positions)
    assert torch.allclose(jacrev(jacrev(scalar))(positions), ref, atol=1e-10)

    # the test suite disables the TorchScript JIT, but torch scripts its
    # forward-mode decompositions on first use
    jit_was_enabled = torch.jit._state._enabled.enabled  # type: ignore
    torch.jit._state.enable()  # type: ignore
    try:
        fwd_rev = jacfwd(jacrev(scalar))(positions)
    finally:
        if not jit_was_enabled:
            torch.jit._state.disable()  # type: ignore
    assert torch.allclose(fwd_rev, ref, atol=1e-10)


@pytest.mark.grad
def test_parameter_gradcheck() -> None:
    """Exponents and coefficients, including the shell normalization."""
    dd: DD = {"dtype": torch.double, "device": DEVICE}
    numbers = samples["H2"]["numbers"].to(DEVICE)
    positions = samples["H2"]["positions"].to(**dd)
    mats = Matrices(numbers, "os", dd)

    alphas = [a.detach().clone().requires_grad_(True) for a in mats.alphas]
    coeffs = [c.detach().clone().requires_grad_(True) for c in mats.coeffs]
    n = len(alphas)

    def func(*params: Tensor) -> Tensor:
        return assemble_matrix(
            mats.kernel,
            mats.ihelp,
            list(params[:n]),
            list(params[n:]),
            positions,
            QUADRUPOLE_COMPONENTS,
        )

    assert torch.autograd.gradcheck(func, (*alphas, *coeffs), atol=1e-6)
