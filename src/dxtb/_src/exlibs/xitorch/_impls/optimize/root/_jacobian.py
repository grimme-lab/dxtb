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

from abc import abstractmethod

import torch

from dxtb._src.exlibs.xitorch.grad import jac

# taking most of the part from SciPy

__all__ = ["BroydenFirst", "BroydenSecond", "LinearMixing"]


class Jacobian:
    """
    Base class for the Jacobians used in rootfinder algorithms.
    """

    @abstractmethod
    def setup(self, x0, y0, func):
        pass

    @abstractmethod
    def solve(self, v, tol: int | float = 0):
        pass

    @abstractmethod
    def update(self, x, y):
        pass


class BroydenFirst(Jacobian):
    """
    Approximating the Jacobian based on Broyden's first approximation.

    [1] B.A. van der Rotten, PhD thesis,
        "A limited memory Broyden method to solve high-dimensional
        systems of nonlinear equations". Mathematisch Instituut,
        Universiteit Leiden, The Netherlands (2003).
    """

    def __init__(self, alpha=None, uv0=None, max_rank=None):
        # The initial guess of inverse Jacobian is `-alpha * I + u v^T`.
        # `max_rank` indicates the maximum rank of the Jacoabian before
        # reducing it
        self.alpha = alpha
        self.uv0 = uv0
        self.max_rank = max_rank

    def setup(self, x0, y0, func):
        self.x_prev = x0
        self.y_prev = y0

        if self.max_rank is None:
            self.max_rank = float("inf")

        if self.alpha is None:
            normy0 = torch.norm(y0)
            ones = torch.ones_like(normy0)
            if normy0:
                self.alpha = 0.5 * torch.max(torch.norm(x0), ones) / normy0
            else:
                self.alpha = ones

        if self.uv0 == "svd":
            self.uv0 = _get_svd_uv0(func, x0)

        # setup the approximate inverse Jacobian
        self.Gm = LowRankMatrix(-self.alpha, self.uv0, "restart")
        # do not uncomment the line below! it causes memory leak
        # I left it here as a lesson for me and us to never repeat this mistake
        # self._reduce = lambda: self.Gm.reduce(self.max_rank)

    def _reduce(self):
        # reduce the size of Gm
        # initially it was a lambda function, but it causes a leak, so
        # I arranged into a method to remove the leak
        self.Gm.reduce(self.max_rank)

    def solve(self, v, tol: int | float = 0):
        res = self.Gm.mv(v)
        return res

    def update(self, x, y):
        dy = y - self.y_prev
        dx = x - self.x_prev  # <-- can become exactly zero
        # update Gm
        self._update(x, y, dx, dy, dx.norm(), dy.norm())

        self.y_prev = y
        self.x_prev = x

    def _update(self, x, y, dx, dy, dxnorm, dynorm):
        # keep the rank small
        self._reduce()

        v = self.Gm.rmv(dx)
        c = dx - self.Gm.mv(dy)
        d = v / torch.dot(dy, v)  # <-- yields NaN
        self.Gm = self.Gm.append(c, d)


class BroydenSecond(BroydenFirst):
    """
    Inverse Jacobian approximation based on Broyden's second method.

    [1] B.A. van der Rotten, PhD thesis,
        "A limited memory Broyden method to solve high-dimensional
        systems of nonlinear equations". Mathematisch Instituut,
        Universiteit Leiden, The Netherlands (2003).
    """

    def _update(self, x, y, dx, dy, dxnorm, dynorm):
        # keep the rank small
        self._reduce()

        v = dy
        c = dx - self.Gm.mv(dy)
        d = v / (dynorm * dynorm)
        self.Gm = self.Gm.append(c, d)


class Anderson(Jacobian):
    """
    https://docs.scipy.org/doc/scipy/reference/optimize.root-anderson.html#optimize-root-anderson
    https://github.com/scipy/scipy/blob/c634fb9c96edde5a800077a764e70d1ef72a4bd2/scipy/optimize/_nonlin.py#L982

    Could use some clean-up and vectorization.
    """

    def __init__(self, alpha=None, uv0=None, max_rank=None, w0=0.01):
        self.alpha = alpha
        self.uv0 = uv0
        self.max_rank = max_rank
        self.dx = []
        self.df = []
        self.w0 = w0

    def setup(self, x0, y0, func):
        self.x_prev = x0
        self.y_prev = y0

        if self.max_rank is None:
            self.max_rank = 5

        if self.alpha is None:
            normy0 = torch.norm(y0)
            ones = torch.ones_like(normy0)
            if normy0:
                self.alpha = 0.5 * torch.max(torch.norm(x0), ones) / normy0
            else:
                self.alpha = ones

    def solve(self, f: torch.Tensor, tol: int | float = 0):
        if self.alpha is None:
            raise RuntimeError("alpha should not be None")
        dx = -self.alpha * f

        n = len(self.dx)
        if n == 0:
            return dx

        df_f = f.new_empty(n)
        for k in range(n):
            df_f[k] = torch.dot(self.df[k], f)

        try:
            gamma = torch.linalg.solve(self.a, df_f)
        except torch.linalg.LinAlgError:
            # singular; reset the Jacobian approximation
            del self.dx[:]
            del self.df[:]
            return dx

        for m in range(n):
            dx += gamma[m] * (self.dx[m] + self.alpha * self.df[m])
        return dx

    def update(self, x, y):
        dy = y - self.y_prev
        dx = x - self.x_prev
        # update Gm
        self._update(x, y, dx, dy, dx.norm(), dy.norm())

        self.y_prev = y
        self.x_prev = x

    def _update(self, x, f: torch.Tensor, dx, df, dx_norm, df_norm):
        if self.max_rank is None:
            raise RuntimeError("max_rank should not be None.")

        if self.max_rank == 0:
            return

        self.dx.append(dx)
        self.df.append(df)

        while len(self.dx) > self.max_rank:
            self.dx.pop(0)
            self.df.pop(0)

        n = len(self.dx)
        a = f.new_zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                a[i, j] = torch.dot(self.df[i], self.df[j])

        offset = a.new_ones(a.shape).fill_diagonal_(1 + self.w0**2)
        a += torch.triu(a * offset, 1).H
        self.a = a


class LinearMixing(Jacobian):
    def __init__(self, alpha=None):
        # The initial guess of inverse Jacobian is ``-alpha * I``
        if alpha is None:
            alpha = -1.0
        self.alpha = alpha

    def setup(self, x0, y0, func):
        pass

    def solve(self, v, tol: int | float = 0):
        return -v * self.alpha

    def update(self, x, y):
        pass


class LowRankMatrix:
    # represents a matrix of `\alpha * I + \sum_n c_n d_n^T`
    def __init__(self, alpha, uv0, reduce_method):
        self.alpha = alpha
        if uv0 is None:
            self.cns = []
            self.dns = []
        else:
            cn0, dn0 = uv0
            self.cns = [cn0]
            self.dns = [dn0]
        self.reduce_method = {"restart": 0, "simple": 1}[reduce_method]

    def mv(self, v):
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.cns[i] * torch.dot(self.dns[i], v)
        return res

    def rmv(self, v):
        res = self.alpha * v
        for i in range(len(self.dns)):
            res += self.dns[i] * torch.dot(self.cns[i], v)
        return res

    def append(self, c, d):
        self.cns.append(c)
        self.dns.append(d)
        if len(self.cns) >= torch.numel(c):
            return FullRankMatrix(self.alpha, self.cns, self.dns)
        return self

    def reduce(self, max_rank, **otherparams):
        if len(self.cns) > max_rank:
            if self.reduce_method == 0:  # restart
                del self.cns[:]
                del self.dns[:]
            elif self.reduce_method == 1:  # simple
                n = len(self.cns)
                del self.cns[: n - max_rank]
                del self.dns[: n - max_rank]


class FullRankMatrix:
    def __init__(self, alpha, cns, dns):
        size = torch.numel(cns[0])
        dtype, device = cns[0].dtype, cns[0].device
        self.mat = torch.eye(size, dtype=dtype, device=device)
        self.mat *= alpha
        for i in range(len(cns)):
            self.mat += torch.ger(cns[i], dns[i])

    def mv(self, v):
        res = torch.matmul(self.mat, v)
        return res

    def rmv(self, v):
        return torch.matmul(self.mat.T, v)

    def append(self, c, d):
        self.mat += torch.ger(c, d)
        return self

    def reduce(self, max_rank, **kwargs):
        pass  # ???


def _get_svd_uv0(func, x0):
    from dxtb._src.exlibs.xitorch.linalg import svd

    # raise RuntimeError
    fjac = jac(func, (x0.clone().requires_grad_(),), idxs=[0])[0]
    # u: (n, 1), s: (1,), vh: (1, n)
    u, s, vh = svd(fjac, k=1, mode="lowest", method="davidson", min_eps=1e-3)
    sinv_sqrt = 1.0 / torch.sqrt(torch.clamp(s, min=0.1))
    uv0 = (sinv_sqrt * vh.squeeze(-2), sinv_sqrt * u.squeeze(-1))
    return uv0
