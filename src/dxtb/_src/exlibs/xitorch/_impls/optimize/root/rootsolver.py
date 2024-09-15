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

# file mostly from SciPy:
# https://github.com/scipy/scipy/blob/914523af3bc03fe7bf61f621363fca27e97ca1d6/scipy/optimize/nonlin.py#L221
# and converted to PyTorch for GPU efficiency

import functools

import torch

from dxtb._src.exlibs.xitorch._impls.optimize.root._jacobian import (
    Anderson,
    BroydenFirst,
    BroydenSecond,
    LinearMixing,
)
from dxtb._src.exlibs.xitorch._utils.exceptions import ConvergenceWarning

__all__ = ["anderson", "broyden1", "broyden2", "linearmixing"]


def _nonlin_solver(
    fcn,
    x0,
    params,
    method,
    # jacobian parameters
    alpha=None,
    uv0=None,
    max_rank=None,
    # stopping criteria
    maxiter=None,
    f_tol=None,
    f_rtol=None,
    x_tol=None,
    x_rtol=None,
    # algorithm parameters
    line_search=True,
    # misc parameters
    verbose=False,
    **unused
):
    """
    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
    uv0: tuple of tensors or str or None
        The initial guess of inverse Jacobian is ``- alpha * I + u v^T``.
        If ``"svd"``, then it uses 1-rank svd to obtain ``u`` and ``v``.
        If None, then ``u`` and ``v`` are zeros.
    max_rank: int or None
        The maximum rank of inverse Jacobian approximation. If ``None``, it
        is ``inf``.
    maxiter: int or None
        Maximum number of iterations, or inf if it is set to None.
    f_tol: float or None
        The absolute tolerance of the norm of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the norm of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    line_search: bool or str
        Options to perform line search. If ``True``, it is set to ``"armijo"``.
    verbose: bool
        Options for verbosity
    """

    if method == "broyden1":
        jacobian = BroydenFirst(alpha=alpha, uv0=uv0, max_rank=max_rank)
    elif method == "broyden2":
        jacobian = BroydenSecond(alpha=alpha, uv0=uv0, max_rank=max_rank)
    elif method in ("linearmixing", "simple"):
        jacobian = LinearMixing(alpha=alpha)
    elif method == "anderson":
        jacobian = Anderson(alpha=alpha, uv0=uv0, max_rank=max_rank)
    else:
        raise RuntimeError("Unknown method: %s" % method)

    if maxiter is None:
        maxiter = 100 * (torch.numel(x0) + 1)
    if line_search is True:
        line_search = "armijo"
    elif line_search is False:
        line_search = None

    # solving complex rootfinder by concatenating real and imaginary part,
    # making the variable twice as long
    x_is_complex = torch.is_complex(x0)

    def _ravel(x: torch.Tensor) -> torch.Tensor:
        # represents x as a long real vector
        if x_is_complex:
            return torch.cat((x.real, x.imag), dim=0).reshape(-1)
        else:
            return x.reshape(-1)

    def _pack(x: torch.Tensor) -> torch.Tensor:
        # pack a long real vector into the shape accepted by fcn
        if x_is_complex:
            n = len(x) // 2
            xreal, ximag = x[:n], x[n:]
            x = xreal + 1j * ximag
        return x.reshape(xshape)

    # shorthand for the function
    xshape = x0.shape
    func = lambda x: _ravel(fcn(_pack(x), *params))
    x = _ravel(x0)

    y = func(x)
    y_norm = y.norm()
    assert isinstance(y_norm, torch.Tensor)

    stop_cond = TerminationCondition(f_tol, f_rtol, y_norm, x_tol, x_rtol)
    if y_norm == 0:
        return x.reshape(xshape)

    # set up the jacobian
    jacobian.setup(x, y, func)

    # solver tolerance
    gamma = 0.9
    eta_max = 0.9999
    eta_threshold = 0.1
    eta = 1e-3

    converge = False
    best_ynorm = y_norm
    best_x = x
    best_dxnorm = x.norm()
    best_iter = 0
    for i in range(maxiter):
        tol = min(eta, eta * y_norm)
        dx = -jacobian.solve(y, tol=tol)

        dx_norm = dx.norm()
        if dx_norm == 0:
            raise ValueError(
                "Jacobian inversion yielded zero vector. "
                "This indicates a bug in the Jacobian "
                "approximation."
            )

        if line_search:
            s, xnew, ynew, y_norm_new = _nonline_line_search(
                func, x, y, dx, search_type=line_search
            )
        else:
            s = 0.95  # modified!!
            xnew = x + s * dx
            ynew = func(xnew)
            y_norm_new = torch.norm(ynew)
            assert isinstance(y_norm_new, torch.Tensor)

        # save the best results
        if y_norm_new < best_ynorm:
            best_x = xnew
            best_dxnorm = dx_norm
            best_ynorm = y_norm_new
            best_iter = i + 1

        jacobian.update(xnew.clone(), ynew)

        # print out dx and df
        to_stop = stop_cond.check(xnew.norm(), y_norm_new, dx_norm)
        if verbose:
            if i < 10 or i % 10 == 0 or to_stop:
                print("%6d: |dx|=%.3e, |f|=%.3e" % (i, dx_norm, y_norm))
        if to_stop:
            converge = True
            break

        # adjust forcing parameters for inexact solve
        eta_A = float(gamma * (y_norm_new / y_norm) ** 2)
        gamma_eta2 = gamma * eta * eta
        if gamma_eta2 < eta_threshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma_eta2))

        y_norm = y_norm_new
        x = xnew
        y = ynew
    if not converge:
        msg = (
            "The rootfinder does not converge after %d iterations. "
            "Best |dx|=%.3e, |f|=%.3e at iter %d"
        ) % (maxiter, best_dxnorm, best_ynorm, best_iter)

        # pylint: disable=import-outside-toplevel
        from dxtb import OutputHandler

        OutputHandler.warn(msg, ConvergenceWarning)
        x = best_x
    return _pack(x)


@functools.wraps(
    _nonlin_solver, assigned=("__annotations__",)
)  # takes only the signature
def anderson(fcn, x0, params=(), **kwargs):
    """
    Anderson acceleration
    """
    return _nonlin_solver(fcn, x0, params, "anderson", **kwargs)


@functools.wraps(
    _nonlin_solver, assigned=("__annotations__",)
)  # takes only the signature
def broyden1(fcn, x0, params=(), **kwargs):
    """
    Solve the root finder or linear equation using the first Broyden method [1]_.
    It can be used to solve minimization by finding the root of the
    function's gradient.

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional systems of nonlinear equations".
           Mathematisch Instituut, Universiteit Leiden, The Netherlands (2003).
           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf
    """
    return _nonlin_solver(fcn, x0, params, "broyden1", **kwargs)


@functools.wraps(
    _nonlin_solver, assigned=("__annotations__",)
)  # takes only the signature
def broyden2(fcn, x0, params=(), **kwargs):
    """
    Solve the root finder or linear equation using the second Broyden method [2]_.
    It can be used to solve minimization by finding the root of the
    function's gradient.

    References
    ----------
    .. [2] B.A. van der Rotten, PhD thesis,
           "A limited memory Broyden method to solve high-dimensional systems of nonlinear equations".
           Mathematisch Instituut, Universiteit Leiden, The Netherlands (2003).
           https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf
    """
    return _nonlin_solver(fcn, x0, params, "broyden2", **kwargs)


def linearmixing(
    fcn,
    x0,
    params=(),
    # jacobian parameters
    alpha=None,
    # stopping criteria
    maxiter=None,
    f_tol=None,
    f_rtol=None,
    x_tol=None,
    x_rtol=None,
    # algorithm parameters
    line_search=True,
    # misc parameters
    verbose=False,
    **unused
):
    """
    Solve the root finding problem by approximating the inverse of Jacobian
    to be a constant scalar.

    Keyword arguments
    -----------------
    alpha: float or None
        The initial guess of inverse Jacobian is ``-alpha * I``.
    maxiter: int or None
        Maximum number of iterations, or inf if it is set to None.
    f_tol: float or None
        The absolute tolerance of the norm of the output ``f``.
    f_rtol: float or None
        The relative tolerance of the norm of the output ``f``.
    x_tol: float or None
        The absolute tolerance of the norm of the input ``x``.
    x_rtol: float or None
        The relative tolerance of the norm of the input ``x``.
    line_search: bool or str
        Options to perform line search. If ``True``, it is set to ``"armijo"``.
    verbose: bool
        Options for verbosity
    """
    kwargs = {
        "alpha": alpha,
        "maxiter": maxiter,
        "f_tol": f_tol,
        "x_tol": x_tol,
        "x_rtol": x_rtol,
        "line_search": line_search,
        "verbose": verbose,
        **unused,
    }
    return _nonlin_solver(fcn, x0, params, "linearmixing", **kwargs)


# set the docstring of the functions
broyden1.__doc__ += _nonlin_solver.__doc__  # type: ignore
broyden2.__doc__ += _nonlin_solver.__doc__  # type: ignore


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return torch.tensor(float("inf"), dtype=v.dtype, device=v.device)
    return torch.norm(v)


def _nonline_line_search(
    func, x, y, dx, search_type="armijo", rdiff=1e-8, smin=1e-2
):
    tmp_s = [0]
    tmp_y = [y]
    tmp_phi = [y.norm() ** 2]

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]
        xt = x + s * dx
        v = func(xt)
        p = _safe_norm(v) ** 2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_y[0] = v
        return p

    # Alternative?
    # s_norm = x.norm() / dx.norm()
    # def derphi(s):
    #     ds = (torch.abs(s) + s_norm + 1) * rdiff
    #     return (phi(s + ds, store=False) - phi(s)) / ds

    if search_type == "armijo":
        s, _ = _scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=smin)

    if s is None:
        # No suitable step length found. Take the full Newton step,
        # and hope for the best.
        s = 1.0

    x = x + s * dx
    if s == tmp_s[0]:
        y = tmp_y[0]
    else:
        y = func(x)
    y_norm = y.norm()

    return s, x, y, y_norm


def _scalar_search_armijo(
    phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0, max_niter=20
):
    phi_a0 = phi(alpha0)
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if phi_a1 <= phi0 + c1 * alpha1 * derphi0:
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    niter = 0
    while (
        alpha1 > amin and niter < max_niter
    ):  # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1 - alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0 * alpha1) - alpha1**2 * (
            phi_a0 - phi0 - derphi0 * alpha0
        )
        a = a / factor
        b = -(alpha0**3) * (
            phi_a1 - phi0 - derphi0 * alpha1
        ) + alpha1**3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (
            3.0 * a
        )
        phi_a2 = phi(alpha2)

        if phi_a2 <= phi0 + c1 * alpha2 * derphi0:
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2
        niter += 1

    # Failed to find a suitable step length
    if niter == max_niter:
        return alpha2, phi_a2
    return None, phi_a1


class TerminationCondition:
    def __init__(self, f_tol, f_rtol, f0_norm, x_tol, x_rtol):
        if f_tol is None:
            f_tol = 1e-6
        if f_rtol is None:
            f_rtol = float("inf")
        if x_tol is None:
            x_tol = 1e-6
        if x_rtol is None:
            x_rtol = float("inf")
        self.f_tol = f_tol
        self.f_rtol = f_rtol
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f0_norm = f0_norm

    def check(self, xnorm, ynorm, dxnorm):
        xtcheck = dxnorm < self.x_tol
        xrcheck = dxnorm < self.x_rtol * xnorm
        ytcheck = ynorm < self.f_tol
        yrcheck = ynorm < self.f_rtol * self.f0_norm
        return xtcheck and xrcheck and ytcheck and yrcheck
