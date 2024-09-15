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
from __future__ import annotations

import warnings
from typing import Mapping, Optional, Tuple, Union

import torch
from tad_mctc.math import einsum

from dxtb.__version__ import __tversion__
from dxtb._src.exlibs.xitorch import LinearOperator
from dxtb._src.exlibs.xitorch._core.linop import MatrixLinearOperator
from dxtb._src.exlibs.xitorch._impls.linalg.symeig import davidson, exacteig
from dxtb._src.exlibs.xitorch._utils.assertfuncs import assert_runtime
from dxtb._src.exlibs.xitorch._utils.exceptions import MathWarning
from dxtb._src.exlibs.xitorch._utils.misc import (
    dummy_context_manager,
    get_and_pop_keys,
    get_method,
    set_default_option,
)
from dxtb._src.exlibs.xitorch.debug.modes import is_debug_enabled
from dxtb._src.exlibs.xitorch.linalg.solve import solve
from dxtb._src.typing import Any, Callable, Tensor

__all__ = ["lsymeig", "usymeig", "symeig", "svd"]


def lsymeig(
    A: LinearOperator,
    neig: Optional[int] = None,
    M: Optional[LinearOperator] = None,
    bck_options: Mapping[str, Any] = {},
    method: Union[str, Callable, None] = None,
    **fwd_options,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return symeig(
        A,
        neig,
        "lowest",
        M,
        method=method,
        bck_options=bck_options,
        **fwd_options,
    )


def usymeig(
    A: LinearOperator,
    neig: Optional[int] = None,
    M: Optional[LinearOperator] = None,
    bck_options: Mapping[str, Any] = {},
    method: Union[str, Callable, None] = None,
    **fwd_options,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return symeig(
        A,
        neig,
        "uppest",
        M,
        method=method,
        bck_options=bck_options,
        **fwd_options,
    )


def symeig(
    A: LinearOperator,
    neig: Optional[int] = None,
    mode: str = "lowest",
    M: Optional[LinearOperator] = None,
    bck_options: Mapping[str, Any] = {},
    method: Union[str, Callable, None] = None,
    **fwd_options,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Obtain ``neig`` lowest eigenvalues and eigenvectors of a linear operator,

    .. math::

        \mathbf{AX = MXE}

    where :math:`\mathbf{A}, \mathbf{M}` are linear operators,
    :math:`\mathbf{E}` is a diagonal matrix containing the eigenvalues, and
    :math:`\mathbf{X}` is a matrix containing the eigenvectors.
    This function can handle derivatives for degenerate cases by setting non-zero
    ``degen_atol`` and ``degen_rtol`` in the backward option using the expressions
    in [1]_.

    Arguments
    ---------
    A: dxtb._src.exlibs.xitorch.LinearOperator
        The linear operator object on which the eigenpairs are constructed.
        It must be a Hermitian linear operator with shape ``(*BA, q, q)``
    neig: int or None
        The number of eigenpairs to be retrieved. If ``None``, all eigenpairs are
        retrieved
    mode: str
        ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
        it will take the lowest ``neig`` eigenpairs.
        If ``"uppest"``, it will take the uppermost ``neig``.
    M: dxtb._src.exlibs.xitorch.LinearOperator
        The transformation on the right hand side. If ``None``, then ``M=I``.
        If specified, it must be a Hermitian with shape ``(*BM, q, q)``.
    bck_options: dict
        Method-specific options for :func:`solve` which used in backpropagation
        calculation with some additional arguments for computing the backward
        derivatives:

        * ``degen_atol`` (``float`` or None): Minimum absolute difference between
          two eigenvalues to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.6``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)
        * ``degen_rtol`` (``float`` or None): Minimum relative difference between
          two eigenvalues to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.4``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)

        Note: the default values of ``degen_atol`` and ``degen_rtol`` are going
        to change in the future. So, for future compatibility, please specify
        the specific values.

    method: str or callable or None
        Method for the eigendecomposition. If ``None``, it will choose
        ``"exacteig"``.
    **fwd_options
        Method-specific options (see method section below).

    Returns
    -------
    tuple of tensors (eigenvalues, eigenvectors)
        It will return eigenvalues and eigenvectors with shapes respectively
        ``(*BAM, neig)`` and ``(*BAM, na, neig)``, where ``*BAM`` is the
        broadcasted shape of ``*BA`` and ``*BM``.

    References
    ----------
    .. [1] Muhammad F. Kasim,
           "Derivatives of partial eigendecomposition of a real symmetric matrix for degenerate cases".
           arXiv:2011.04366 (2020)
           `https://arxiv.org/abs/2011.04366 <https://arxiv.org/abs/2011.04366>`_
    """
    assert_runtime(A.is_hermitian, "The linear operator A must be Hermitian")
    assert_runtime(
        not torch.is_grad_enabled() or A.is_getparamnames_implemented,
        "The _getparamnames(self, prefix) of linear operator A must be "
        "implemented if using symeig with grad enabled",
    )
    if M is not None:
        assert_runtime(
            M.is_hermitian, "The linear operator M must be Hermitian"
        )
        assert_runtime(
            M.shape[-1] == A.shape[-1],
            f"The shape of A & M must match (A: {A.shape}, M: {M.shape})",
        )
        assert_runtime(
            not torch.is_grad_enabled() or M.is_getparamnames_implemented,
            "The _getparamnames(self, prefix) of linear operator M must be "
            "implemented if using symeig with grad enabled",
        )
    mode = mode.lower()
    if mode == "uppermost":
        mode = "uppest"
    if method is None:
        if isinstance(A, MatrixLinearOperator) and (
            M is None or isinstance(M, MatrixLinearOperator)
        ):
            method = "exacteig"
        else:
            # TODO: implement robust LOBPCG and put it here
            method = "exacteig"
    if neig is None:
        neig = A.shape[-1]

    # perform expensive check if debug mode is enabled
    if is_debug_enabled():
        A.check()
        if M is not None:
            M.check()

    if method == "exacteig":
        return exacteig(A, neig, mode, M)
    else:
        fwd_options["method"] = method
        # get the unique parameters of A & M
        params = A.getlinopparams()
        mparams = M.getlinopparams() if M is not None else []
        na = len(params)
        return symeig_torchfcn(
            A, neig, mode, M, fwd_options, bck_options, na, *params, *mparams
        )


def svd(
    A: LinearOperator,
    k: Optional[int] = None,
    mode: str = "uppest",
    bck_options: Mapping[str, Any] = {},
    method: Union[str, Callable, None] = None,
    **fwd_options,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Perform the singular value decomposition (SVD):

    .. math::

        \mathbf{A} = \mathbf{U\Sigma V}^H

    where :math:`\mathbf{U}` and :math:`\mathbf{V}` are semi-unitary matrix and
    :math:`\mathbf{\Sigma}` is a diagonal matrix containing real non-negative
    numbers.
    This function can handle derivatives for degenerate singular values by setting non-zero
    ``degen_atol`` and ``degen_rtol`` in the backward option using the expressions
    in [1]_.

    Arguments
    ---------
    A: dxtb._src.exlibs.xitorch.LinearOperator
        The linear operator to be decomposed. It has a shape of ``(*BA, m, n)``
        where ``(*BA)`` is the batched dimension of ``A``.
    k: int or None
        The number of decomposition obtained. If ``None``, it will be
        ``min(*A.shape[-2:])``
    mode: str
        ``"lowest"`` or ``"uppermost"``/``"uppest"``. If ``"lowest"``,
        it will take the lowest ``k`` decomposition.
        If ``"uppest"``, it will take the uppermost ``k``.
    bck_options: dict
        Method-specific options for :func:`solve` which used in backpropagation
        calculation with some additional arguments for computing the backward
        derivatives:

        * ``degen_atol`` (``float`` or None): Minimum absolute difference between
          two singular values to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.6``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)
        * ``degen_rtol`` (``float`` or None): Minimum relative difference between
          two singular values to be treated as degenerate. If None, it is
          ``torch.finfo(dtype).eps**0.4``. If 0.0, no special treatment on
          degeneracy is applied. (default: None)

        Note: the default values of ``degen_atol`` and ``degen_rtol`` are going
        to change in the future. So, for future compatibility, please specify
        the specific values.

    method: str or callable or None
        Method for the svd (same options for :func:`symeig`). If ``None``,
        it will choose ``"exacteig"``.
    **fwd_options
        Method-specific options (see method section below).

    Returns
    -------
    tuple of tensors (u, s, vh)
        It will return ``u, s, vh`` with shapes respectively
        ``(*BA, m, k)``, ``(*BA, k)``, and ``(*BA, k, n)``.

    Note
    ----
    It is a naive implementation of symmetric eigendecomposition of ``A.H @ A``
    or ``A @ A.H`` (depending which one is cheaper)

    References
    ----------
    .. [1] Muhammad F. Kasim,
           "Derivatives of partial eigendecomposition of a real symmetric matrix for degenerate cases".
           arXiv:2011.04366 (2020)
           `https://arxiv.org/abs/2011.04366 <https://arxiv.org/abs/2011.04366>`_
    """
    # A: (*BA, m, n)
    # adapted from scipy.sparse.linalg.svds

    if is_debug_enabled():
        A.check()

    m = A.shape[-2]
    n = A.shape[-1]
    if m < n:
        AAsym = A.matmul(A.H, is_hermitian=True)
    else:
        AAsym = A.H.matmul(A, is_hermitian=True)

    eivals, eivecs = symeig(
        AAsym, k, mode, bck_options=bck_options, method=method, **fwd_options
    )  # (*BA, k) and (*BA, min(mn), k)

    # clamp the eigenvalues to a small positive values to avoid numerical
    # instability
    eivals = torch.clamp(eivals, min=0.0)
    s = torch.sqrt(eivals)  # (*BA, k)
    sdiv = torch.clamp(s, min=1e-12).unsqueeze(-2)  # (*BA, 1, k)
    if m < n:
        u = eivecs  # (*BA, m, k)
        v = A.rmm(u) / sdiv  # (*BA, n, k)
    else:
        v = eivecs  # (*BA, n, k)
        u = A.mm(v) / sdiv  # (*BA, m, k)
    vh = v.transpose(-2, -1).conj()
    return u, s, vh


class SymeigMethodBase(torch.autograd.Function):

    @staticmethod
    def backward(ctx, grad_evals, grad_evecs):
        # grad_evals: (*BAM, neig)
        # grad_evecs: (*BAM, na, neig)

        # get the variables from ctx
        evals, evecs = ctx.saved_tensors[:2]
        na = ctx.na
        amparams = ctx.saved_tensors[2:]
        params = amparams[:na]
        mparams = amparams[na:]

        M = ctx.M
        A = ctx.A
        degen_atol: float = ctx.bck_alg_config["degen_atol"]
        degen_rtol: float = ctx.bck_alg_config["degen_rtol"]

        # set the default values of degen_*tol
        dtype = evals.dtype
        if degen_atol is None:
            degen_atol = torch.finfo(dtype).eps ** 0.6
        if degen_rtol is None:
            degen_rtol = torch.finfo(dtype).eps ** 0.4

        # check the degeneracy
        if degen_atol > 0 or degen_rtol > 0:
            # idx_degen: (*BAM, neig, neig)
            idx_degen, isdegenerate = _check_degen(
                evals, degen_atol, degen_rtol
            )
        else:
            isdegenerate = False
        if not isdegenerate:
            idx_degen = None

        # the loss function where the gradient will be retrieved
        # warnings: if not all params have the connection to the output of A,
        # it could cause an infinite loop because pytorch will keep looking
        # for the *params node and propagate further backward via the `evecs`
        # path. So make sure all the *params are all connected in the graph.
        with torch.enable_grad():
            params = [p.clone().requires_grad_() for p in params]
            with A.uselinopparams(*params):
                loss = A.mm(evecs)  # (*BAM, na, neig)

        # if degenerate, check the conditions for finite derivative
        if is_debug_enabled() and isdegenerate:
            xtg = torch.matmul(evecs.transpose(-2, -1).conj(), grad_evecs)
            req1 = idx_degen * (xtg - xtg.transpose(-2, -1).conj())
            reqtol = (
                xtg.abs().max()
                * grad_evecs.shape[-2]
                * torch.finfo(grad_evecs.dtype).eps
            )

            if not torch.all(torch.abs(req1) <= reqtol):
                # if the requirements are not satisfied, raises a warning
                msg = (
                    "Degeneracy appears but the loss function seem to depend "
                    "strongly on the eigenvector. The gradient might be incorrect.\n"
                )
                msg += "Eigenvalues:\n%s\n" % str(evals)
                msg += "Degenerate map:\n%s\n" % str(idx_degen)
                msg += "Requirements (should be all 0s):\n%s" % str(req1)
                warnings.warn(MathWarning(msg))

        # calculate the contributions from the eigenvalues
        gevalsA = grad_evals.unsqueeze(-2) * evecs  # (*BAM, na, neig)

        # calculate the contributions from the eigenvectors
        with (
            M.uselinopparams(*mparams)
            if M is not None
            else dummy_context_manager()
        ):
            # orthogonalize the grad_evecs with evecs
            B = _ortho(grad_evecs, evecs, D=idx_degen, M=M, mright=False)

            # Based on test cases, complex datatype is more likely to suffer from
            # singularity error when doing the inverse. Therefore, I add a small
            # offset here to prevent that from happening
            if torch.is_complex(B):
                evals_offset = evals + 1e-14
            else:
                evals_offset = evals

            with A.uselinopparams(*params):
                gevecs = solve(
                    A,
                    -B,
                    evals_offset,
                    M,
                    bck_options=ctx.bck_config,
                    **ctx.bck_config,
                )  # (*BAM, na, neig)

            # orthogonalize gevecs w.r.t. evecs
            gevecsA = _ortho(gevecs, evecs, D=None, M=M, mright=True)

        # accummulate the gradient contributions
        gaccumA = gevalsA + gevecsA
        grad_params = torch.autograd.grad(
            outputs=(loss,),
            inputs=params,
            grad_outputs=(gaccumA,),
            create_graph=torch.is_grad_enabled(),
        )

        grad_mparams = []
        if ctx.M is not None:
            with torch.enable_grad():
                mparams = [p.clone().requires_grad_() for p in mparams]
                with M.uselinopparams(*mparams):
                    mloss = M.mm(evecs)  # (*BAM, na, neig)
            gevalsM = -gevalsA * evals.unsqueeze(-2)
            gevecsM = -gevecsA * evals.unsqueeze(-2)

            # the contribution from the parallel elements
            gevecsM_par = (
                -0.5 * einsum("...ae,...ae->...e", grad_evecs, evecs.conj())
            ).unsqueeze(
                -2
            ) * evecs  # (*BAM, na, neig)

            gaccumM = gevalsM + gevecsM + gevecsM_par
            grad_mparams = torch.autograd.grad(
                outputs=(mloss,),
                inputs=mparams,
                grad_outputs=(gaccumM,),
                create_graph=torch.is_grad_enabled(),
            )

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            *grad_params,
            *grad_mparams,
        )


class SymeigMethod_V1(SymeigMethodBase):

    @staticmethod
    def forward(ctx, A, neig, mode, M, fwd_options, bck_options, na, *amparams):
        # A: LinearOperator (*BA, q, q)
        # M: LinearOperator (*BM, q, q) or None

        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({}, fwd_options)
        ctx.bck_config = set_default_option(
            {
                "degen_atol": None,
                "degen_rtol": None,
            },
            bck_options,
        )

        # options for calculating the backward (not for `solve`)
        alg_keys = ["degen_atol", "degen_rtol"]
        ctx.bck_alg_config = get_and_pop_keys(ctx.bck_config, alg_keys)

        method = config.pop("method")
        with A.uselinopparams(*params), (
            M.uselinopparams(*mparams)
            if M is not None
            else dummy_context_manager()
        ):
            methods = {
                "davidson": davidson,
                "custom_exacteig": custom_exacteig,
            }
            method_fcn = get_method("symeig", methods, method)
            evals, evecs = method_fcn(A, neig, mode, M, **config)

        # save for the backward
        # evals: (*BAM, neig)
        # evecs: (*BAM, na, neig)
        ctx.save_for_backward(evals, evecs, *amparams)
        ctx.na = na
        ctx.A = A
        ctx.M = M
        return evals, evecs


class SymeigMethod_V2(SymeigMethodBase):

    @staticmethod
    def forward(A, neig, mode, M, fwd_options, bck_options, na, *amparams):
        # A: LinearOperator (*BA, q, q)
        # M: LinearOperator (*BM, q, q) or None

        # separate the sets of parameters
        params = amparams[:na]
        mparams = amparams[na:]

        config = set_default_option({}, fwd_options)

        method = config.pop("method")
        with A.uselinopparams(*params), (
            M.uselinopparams(*mparams)
            if M is not None
            else dummy_context_manager()
        ):
            methods = {
                "davidson": davidson,
                "custom_exacteig": custom_exacteig,
            }
            method_fcn = get_method("symeig", methods, method)
            evals, evecs = method_fcn(A, neig, mode, M, **config)

        return evals, evecs

    @staticmethod
    def setup_context(ctx, inputs: tuple, output: tuple[Tensor, Tensor]):
        A, _, _, M, _, bck_options, na, amparams = inputs
        evals, evecs = output

        ctx.bck_config = set_default_option(
            {
                "degen_atol": None,
                "degen_rtol": None,
            },
            bck_options,
        )

        # options for calculating the backward (not for `solve`)
        alg_keys = ["degen_atol", "degen_rtol"]
        ctx.bck_alg_config = get_and_pop_keys(ctx.bck_config, alg_keys)

        # save for the backward
        # evals: (*BAM, neig)
        # evecs: (*BAM, na, neig)
        ctx.save_for_backward(evals, evecs, *amparams)
        ctx.na = na
        ctx.A = A
        ctx.M = M


def symeig_torchfcn(
    A, neig, mode, M, fwd_options, bck_options, na, *amparams
) -> tuple[Tensor, Tensor]:

    SymeigMethod = (
        SymeigMethod_V1 if __tversion__ < (2, 0, 0) else SymeigMethod_V2
    )
    res = SymeigMethod.apply(
        A, neig, mode, M, fwd_options, bck_options, na, *amparams
    )
    assert res is not None
    return res[0], res[1]


def _check_degen(
    evals: torch.Tensor, degen_atol: float, degen_rtol: float
) -> Tuple[torch.Tensor, bool]:
    # evals: (*BAM, neig)

    # (*BAM, neig, neig)
    evals_diff = torch.abs(evals.unsqueeze(-2) - evals.unsqueeze(-1))
    degen_thrsh = degen_atol + degen_rtol * torch.abs(evals).unsqueeze(-1)
    idx_degen = (evals_diff < degen_thrsh).to(evals.dtype)
    isdegenerate = bool(torch.sum(idx_degen) > torch.numel(evals))
    return idx_degen, isdegenerate


def _ortho(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    D: Optional[torch.Tensor] = None,
    M: Optional[LinearOperator] = None,
    mright: bool = False,
) -> torch.Tensor:
    # orthogonalize every column in A w.r.t. columns in B
    # D is the degeneracy map, if None, it is identity matrix
    # M is the overlap matrix (in LinearOperator)
    # mright indicates whether to operate M at the right or at the left

    # shapes:
    # A: (*BAM, na, neig)
    # B: (*BAM, na, neig)
    if D is None:
        # contracted using opt_einsum
        str1 = "...rc,...rc->...c"
        Bconj = B.conj()
        if M is None:
            return A - einsum(str1, A, Bconj).unsqueeze(-2) * B
        elif mright:
            return A - einsum(str1, M.mm(A), Bconj).unsqueeze(-2) * B
        else:
            return A - M.mm(einsum(str1, A, Bconj).unsqueeze(-2) * B)
    else:
        BH = B.transpose(-2, -1).conj()
        if M is None:
            DBHA = D * torch.matmul(BH, A)
            return A - torch.matmul(B, DBHA)
        elif mright:
            DBHA = D * torch.matmul(BH, M.mm(A))
            return A - torch.matmul(B, DBHA)
        else:
            DBHA = D * torch.matmul(BH, A)
            return A - M.mm(torch.matmul(B, DBHA))


def custom_exacteig(A, neig, mode, M=None, **options):
    return exacteig(A, neig, mode, M)
