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
import warnings
from typing import Any, Callable, Mapping, Optional, Union

import torch
from tad_mctc.math import einsum

from dxtb.__version__ import __tversion__
from dxtb._src.exlibs.xitorch import LinearOperator
from dxtb._src.exlibs.xitorch._core.linop import MatrixLinearOperator
from dxtb._src.exlibs.xitorch._impls.linalg.solve import (
    _get_batchdims,
    bicgstab,
    broyden1_solve,
    cg,
    exactsolve,
    gmres,
    wrap_gmres,
)
from dxtb._src.exlibs.xitorch._utils.assertfuncs import assert_runtime
from dxtb._src.exlibs.xitorch._utils.misc import (
    dummy_context_manager,
    get_method,
    set_default_option,
)
from dxtb._src.exlibs.xitorch.debug.modes import is_debug_enabled

__all__ = ["solve"]


def solve(
    A: LinearOperator,
    B: torch.Tensor,
    E: Union[torch.Tensor, None] = None,
    M: Optional[LinearOperator] = None,
    bck_options: Mapping[str, Any] = {},
    method: Union[str, Callable, None] = None,
    **fwd_options,
) -> torch.Tensor:
    r"""
    Performing iterative method to solve the equation

    .. math::

        \mathbf{AX=B}

    or

    .. math::

        \mathbf{AX-MXE=B}

    where :math:`\mathbf{E}` is a diagonal matrix.
    This function can also solve batched multiple inverse equation at the
    same time by applying :math:`\mathbf{A}` to a tensor :math:`\mathbf{X}`
    with shape ``(...,na,ncols)``.
    The applied :math:`\mathbf{E}` are not necessarily identical for each column.

    Arguments
    ---------
    A: dxtb._src.exlibs.xitorch.LinearOperator
        A linear operator that takes an input ``X`` and produce the vectors in the same
        space as ``B``.
        It should have the shape of ``(*BA, na, na)``
    B: torch.Tensor
        The tensor on the right hand side with shape ``(*BB, na, ncols)``
    E: torch.Tensor or None
        If a tensor, it will solve :math:`\mathbf{AX-MXE = B}`.
        It will be regarded as the diagonal of the matrix.
        Otherwise, it just solves :math:`\mathbf{AX = B}` and ``M`` is ignored.
        If it is a tensor, it should have shape of ``(*BE, ncols)``.
    M: dxtb._src.exlibs.xitorch.LinearOperator or None
        The transformation on the ``E`` side. If ``E`` is ``None``,
        then this argument is ignored.
        If E is not ``None`` and ``M`` is ``None``, then ``M=I``.
        If LinearOperator, it must be Hermitian with shape ``(*BM, na, na)``.
    bck_options: dict
        Options of the iterative solver in the backward calculation.
    method: str or callable or None
        The method of linear equation solver. If ``None``, it will choose
        ``"cg"`` or ``"bicgstab"`` based on the matrices symmetry.
        `Note`: default method will be changed quite frequently, so if you want
        future compatibility, please specify a method.
    **fwd_options
        Method-specific options (see method below)

    Returns
    -------
    torch.Tensor
        The tensor :math:`\mathbf{X}` that satisfies :math:`\mathbf{AX-MXE=B}`.
    """
    assert_runtime(
        A.shape[-1] == A.shape[-2],
        "The linear operator A must have a square shape",
    )
    assert_runtime(
        A.shape[-1] == B.shape[-2],
        f"Mismatch shape of A & B (A: {A.shape}, B: {B.shape})",
    )
    assert_runtime(
        not torch.is_grad_enabled() or A.is_getparamnames_implemented,
        "The _getparamnames(self, prefix) of linear operator A must be "
        "implemented if using solve with grad enabled",
    )
    if M is not None:
        assert_runtime(
            M.shape[-1] == M.shape[-2],
            "The linear operator M must have a square shape",
        )
        assert_runtime(
            M.shape[-1] == A.shape[-1],
            f"The shape of A & M must match (A: {A.shape}, M: {M.shape})",
        )
        assert_runtime(
            M.is_hermitian, "The linear operator M must be a Hermitian matrix"
        )
        assert_runtime(
            not torch.is_grad_enabled() or M.is_getparamnames_implemented,
            "The _getparamnames(self, prefix) of linear operator M must be "
            "implemented if using solve with grad enabled",
        )
    if E is not None:
        assert_runtime(
            E.shape[-1] == B.shape[-1],
            "The last dimension of E & B must match (E: %s, B: %s)"
            % (E.shape, B.shape),
        )
    if E is None and M is not None:
        warnings.warn(
            "M is supplied but will be ignored because E is not supplied"
        )

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if method is None:
        if isinstance(A, MatrixLinearOperator) and (
            M is None or isinstance(M, MatrixLinearOperator)
        ):
            method = "exactsolve"
        elif A.shape[-1] <= 5:  # for small matrix
            method = "exactsolve"
        else:
            is_hermit = A.is_hermitian and (M is None or M.is_hermitian)
            method = "cg" if is_hermit else "bicgstab"

    if method == "exactsolve":
        return exactsolve(A, B, E, M)
    else:
        # get the unique parameters of A
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return solve_torchfcn(
            A, B, E, M, method, fwd_options, bck_options, na, *params, *mparams
        )


def solve_torchfcn(
    A, B, E, M, method, fwd_options, bck_options, na, *allparams
) -> torch.Tensor:
    Solver = Solver_V1 if __tversion__ < (2, 0, 0) else Solver_V2
    res = Solver.apply(
        A, B, E, M, method, fwd_options, bck_options, na, *allparams
    )
    assert res is not None
    return res


class SolverBase(torch.autograd.Function):

    @staticmethod
    def backward(ctx, grad_x):
        # grad_x: (*BABEM, nr, ncols)
        # x: (*BABEM, nr, ncols)
        x = ctx.saved_tensors[0]
        idx_all_params = 1 if ctx.e_is_none else 2
        all_params = ctx.saved_tensors[idx_all_params:]
        params = all_params[: ctx.na]
        mparams = all_params[ctx.na :]
        E = None if ctx.e_is_none else ctx.saved_tensors[1]

        # solve (A-biases*M)^T v = grad_x
        # this is the grad of B
        with ctx.A.uselinopparams(*params), (
            ctx.M.uselinopparams(*mparams)
            if ctx.M is not None
            else dummy_context_manager()
        ):
            AT = ctx.A.H  # (*BA, nr, nr)
            MT = ctx.M.H if ctx.M is not None else None  # (*BM, nr, nr)
            Econj = E.conj() if E is not None else None
            v = solve(
                AT,
                grad_x,
                Econj,
                MT,
                bck_options=ctx.bck_config,
                **ctx.bck_config,
            )  # (*BABEM, nr, ncols)
        grad_B = v

        # calculate the grad of matrices parameters
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in params]
            with ctx.A.uselinopparams(*params):
                loss = -ctx.A.mm(x)  # (*BABEM, nr, ncols)

        grad_params = torch.autograd.grad(
            (loss,),
            params,
            grad_outputs=(v,),
            create_graph=torch.is_grad_enabled(),
            allow_unused=True,
        )

        # calculate the biases gradient
        grad_E = None
        if E is not None:
            if ctx.M is None:
                Mx = x
            else:
                with ctx.M.uselinopparams(*mparams):
                    Mx = ctx.M.mm(x)  # (*BABEM, nr, ncols)
            grad_E = einsum(
                "...rc,...rc->...c", v, Mx.conj()
            )  # (*BABEM, ncols)

        # calculate the gradient to the biases matrices
        grad_mparams = []
        if ctx.M is not None and E is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in mparams]
                lmbdax = x * E.unsqueeze(-2)
                with ctx.M.uselinopparams(*mparams):
                    mloss = ctx.M.mm(lmbdax)

            grad_mparams = torch.autograd.grad(
                (mloss,),
                mparams,
                grad_outputs=(v,),
                create_graph=torch.is_grad_enabled(),
                allow_unused=True,
            )

        return (
            None,
            grad_B,
            grad_E,
            None,
            None,
            None,
            None,
            None,
            *grad_params,
            *grad_mparams,
        )


class Solver_V1(SolverBase):
    @staticmethod
    def forward(
        ctx, A, B, E, M, method, fwd_options, bck_options, na, *all_params
    ):
        # A: (*BA, nr, nr)
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols) or None
        # M: (*BM, nr, nr) or None
        # all_params: list of tensor of any shape
        # returns: (*BABEM, nr, ncols)

        # separate the parameters for A and for M
        params = all_params[:na]
        mparams = all_params[na:]

        config = set_default_option({}, fwd_options)
        ctx.bck_config = set_default_option({}, bck_options)

        if torch.all(B == 0):  # special case
            dims = (*_get_batchdims(A, B, E, M), *B.shape[-2:])
            x = torch.zeros(dims, dtype=B.dtype, device=B.device)
        else:
            with A.uselinopparams(*params), (
                M.uselinopparams(*mparams)
                if M is not None
                else dummy_context_manager()
            ):
                methods = {
                    "custom_exactsolve": custom_exactsolve,
                    "scipy_gmres": wrap_gmres,
                    "broyden1": broyden1_solve,
                    "cg": cg,
                    "bicgstab": bicgstab,
                    "gmres": gmres,
                }
                method_fcn = get_method("solve", methods, method)
                x = method_fcn(A, B, E, M, **config)

        ctx.e_is_none = E is None
        ctx.A = A
        ctx.M = M
        if ctx.e_is_none:
            ctx.save_for_backward(x, *all_params)
        else:
            ctx.save_for_backward(x, E, *all_params)
        ctx.na = na
        return x


class Solver_V2(SolverBase):

    generate_vmap_rule = True

    @staticmethod
    def setup_context(ctx, inputs: tuple, outputs: torch.Tensor):
        A = inputs[0]
        # B = inputs[1]
        E = inputs[2]
        M = inputs[3]
        # method = inputs[4]
        # fwd_options = inputs[5]
        bck_options = inputs[6]
        na = inputs[7]
        all_params = inputs[8:]

        ctx.bck_config = set_default_option({}, bck_options)
        ctx.e_is_none = E is None
        ctx.na = na
        ctx.A = A
        ctx.M = M

        if ctx.e_is_none:
            ctx.save_for_backward(outputs, *all_params)
        else:
            ctx.save_for_backward(outputs, E, *all_params)

    @staticmethod
    def forward(A, B, E, M, method, fwd_options, bck_options, na, *all_params):
        # A: (*BA, nr, nr)
        # B: (*BB, nr, ncols)
        # E: (*BE, ncols) or None
        # M: (*BM, nr, nr) or None
        # all_params: list of tensor of any shape
        # returns: (*BABEM, nr, ncols)

        # separate the parameters for A and for M
        params = all_params[:na]
        mparams = all_params[na:]

        config = set_default_option({}, fwd_options)

        if torch.all(B == 0):  # special case
            dims = (*_get_batchdims(A, B, E, M), *B.shape[-2:])
            x = torch.zeros(dims, dtype=B.dtype, device=B.device)
        else:
            with A.uselinopparams(*params), (
                M.uselinopparams(*mparams)
                if M is not None
                else dummy_context_manager()
            ):
                methods = {
                    "custom_exactsolve": custom_exactsolve,
                    "scipy_gmres": wrap_gmres,
                    "broyden1": broyden1_solve,
                    "cg": cg,
                    "bicgstab": bicgstab,
                    "gmres": gmres,
                }
                method_fcn = get_method("solve", methods, method)
                x = method_fcn(A, B, E, M, **config)

        return x


def custom_exactsolve(A, B, E=None, M=None, **options):
    # A: (*BA, na, na)
    # B: (*BB, na, ncols)
    # E: (*BE, ncols)
    # M: (*BM, na, na)
    return exactsolve(A, B, E, M)
