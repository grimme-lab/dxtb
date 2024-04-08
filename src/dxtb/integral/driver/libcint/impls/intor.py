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
Integral Interface
==================

Interface for libcint to calculate the integrals.
"""

from __future__ import annotations

import copy
import ctypes
import operator
from functools import reduce

import numpy as np
import torch
from tad_mctc.convert import numpy_to_tensor
from tad_mctc.math import einsum

from dxtb.__version__ import __tversion__
from dxtb.typing import Callable, Protocol, Tensor

from .namemanager import IntorNameManager
from .utils import int2ctypes, np2ctypes
from .wrapper import LibcintWrapper

try:
    from dxtblibs import CGTO, CINT
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules. {e}. {e.name} provides a Python "
        "interface to the 'libcint' library for fast integral evaluation. "
        "It can be installed via 'pip install dxtblibs'."
    ) from e

__all__ = ["int1e", "overlap"]


def _int2c(
    allcoeffs: Tensor,
    allalphas: Tensor,
    allposs: Tensor,
    wrappers: list[LibcintWrapper],
    namemgr: IntorNameManager,
    hermitian: bool = False,
) -> Tensor:
    """
    Calculate the 2-centre integrals.

    Parameters
    ----------
    allcoeffs : Tensor
        All coefficients of the basis functions.
    allalphas : Tensor
        All exponents of the basis functions.
    allposs : Tensor
        All atomic positions of the basis functions.
    wrappers : list[LibcintWrapper]
        List of wrappers for the integrals.
    namemgr : IntorNameManager
        Name manager for the integrals.
    hermitian : bool, optional
        Whether the integral is hermitian. Defaults to `False`.

    Returns
    -------
    Tensor
        Integral tensor.
    """
    Int2cFunction = Int2c_V1 if __tversion__ < (2, 0, 0) else Int2c_V2

    integral = Int2cFunction.apply(
        allcoeffs, allalphas, allposs, wrappers, namemgr, hermitian
    )

    # only for typing
    assert integral is not None
    return integral


def _check_and_set(
    wrapper: LibcintWrapper, other: LibcintWrapper | None = None
) -> LibcintWrapper:
    """
    Check the wrapper and set the default value of "other" in the integrals.

    Parameters
    ----------
    wrapper : LibcintWrapper
        Wrapper for the integrals.
    other : LibcintWrapper | None, optional
        Possibly other wrapper. Defaults to `None`.

    Returns
    -------
    LibcintWrapper
        The (checked) other wrapper.
    """
    if other is not None:
        atm0, bas0, env0 = wrapper.atm_bas_env
        atm1, bas1, env1 = other.atm_bas_env
        msg = (
            "Argument `other*` does not have the same parent as the wrapper. "
            "Please do `LibcintWrapper.concatenate` on those wrappers first."
        )
        assert id(atm0) == id(atm1), msg
        assert id(bas0) == id(bas1), msg
        assert id(env0) == id(env1), msg
    else:
        other = wrapper
    assert isinstance(other, LibcintWrapper)
    return other


def int1e(
    shortname: str,
    wrapper: LibcintWrapper,
    other: LibcintWrapper | None = None,
    hermitian: bool = False,
) -> Tensor:
    """
    Shortcut for the 2-centre 1-electron integrals.

    Parameters
    ----------
    shortname : str
        Short name of the integral.
    wrapper : LibcintWrapper
        Interface for libcint.
    other : LibcintWrapper | None, optional
        The "other" interface for libcint. Defaults to `None`.
    hermitian : bool, optional
        Explicitly request the hermitian integral. Defaults to `False`.

    Returns
    -------
    Tensor
        Integral tensor.
    """
    # check and set the other parameters
    other1 = _check_and_set(wrapper, other)

    return _int2c(
        *wrapper.params,
        wrappers=[wrapper, other1],
        namemgr=IntorNameManager("int1e", shortname),
        hermitian=hermitian,
    )


def overlap(wrapper: LibcintWrapper, other: LibcintWrapper | None = None) -> Tensor:
    """
    Shortcut for the overlap integral.

    Parameters
    ----------
    wrapper : LibcintWrapper
        Interface for libcint.
    other : LibcintWrapper | None, optional
        Interface for libcint. Defaults to `None`.

    Returns
    -------
    Tensor
        Overlap integral.
    """
    return int1e("ovlp", wrapper, other=other, hermitian=True)


class CTX(Protocol):
    save_for_backward: Callable[[Tensor, Tensor, Tensor], None]
    saved_tensors: tuple[Tensor, Tensor, Tensor]
    wrappers: list[LibcintWrapper]
    int_nmgr: IntorNameManager
    hermitian: bool


class BaseInt2c(torch.autograd.Function):
    """
    Base class for version-specific autograd function for 2-centre integrals.
    Different PyTorch versions only require different `forward()` signatures.
    """

    @staticmethod
    def backward(ctx: CTX, grad_out: Tensor) -> tuple[Tensor | None, ...]:
        # grad_out: (..., nao0, nao1)
        allcoeffs = ctx.saved_tensors[0]
        allalphas = ctx.saved_tensors[1]
        allposs = ctx.saved_tensors[2]
        wrappers = ctx.wrappers
        int_nmgr = ctx.int_nmgr
        hermitian = ctx.hermitian

        # gradient for all atomic positions
        grad_allposs: Tensor | None = None
        if allposs.requires_grad:
            grad_allposs = torch.zeros_like(allposs)  # (natom, ndim)
            grad_allpossT = torch.zeros_like(allposs).transpose(-2, -1)  # (ndim, natom)

            # get the integrals required for the derivatives
            sname_derivs = [int_nmgr.get_intgl_deriv_namemgr("ip", ib) for ib in (0, 1)]
            # new axes added to the dimension
            new_axes_pos = [
                int_nmgr.get_intgl_deriv_newaxispos("ip", ib) for ib in (0, 1)
            ]
            assert isinstance(new_axes_pos[0], int) and isinstance(new_axes_pos[1], int)

            def int_fcn(
                wrappers: list[LibcintWrapper], namemgr: IntorNameManager
            ) -> Tensor:
                return _int2c(
                    allcoeffs, allalphas, allposs, wrappers, namemgr, hermitian
                )

            # list of tensors with shape: (ndim, ..., nao0, nao1)
            dout_dposs = _get_integrals(sname_derivs, wrappers, int_fcn, new_axes_pos)

            ndim = dout_dposs[0].shape[0]
            shape = (ndim, -1, *dout_dposs[0].shape[-2:])
            grad_out2 = grad_out.reshape(shape[1:])

            # negative because the integral calculates the nabla w.r.t. the
            # spatial coordinate, not the basis central position
            grad_dpos_i = -einsum(
                "sij,dsij->di", grad_out2, dout_dposs[0].reshape(shape)
            )
            grad_dpos_j = -einsum(
                "sij,dsij->dj", grad_out2, dout_dposs[1].reshape(shape)
            )

            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)

            # grad_allpossT is only a view of grad_allposs, so the operation
            # below also changes grad_allposs
            # grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom0, src=grad_dpos_i)
            # grad_allpossT.scatter_add_(dim=-1, index=ao_to_atom1, src=grad_dpos_j)

            updated_grad_allpossT = torch.scatter_add(
                grad_allpossT, dim=-1, index=ao_to_atom0, src=grad_dpos_i
            )
            updated_grad_allpossT = torch.scatter_add(
                updated_grad_allpossT, dim=-1, index=ao_to_atom1, src=grad_dpos_j
            )

            # Transpose back to match the shape of grad_allposs
            grad_allposs = updated_grad_allpossT.transpose(-2, -1)

        # gradient for the basis coefficients
        grad_allcoeffs: Tensor | None = None
        grad_allalphas: Tensor | None = None
        if allcoeffs.requires_grad or allalphas.requires_grad:
            # obtain the uncontracted wrapper and mapping
            # uao2aos: list of (nu_ao0,), (nu_ao1,)
            u_wrappers_tup, uao2aos_tup = zip(
                *[w.get_uncontracted_wrapper() for w in wrappers]
            )
            u_wrappers = list(u_wrappers_tup)
            uao2aos = list(uao2aos_tup)
            u_params = u_wrappers[0].params

            # get the uncontracted (gathered) grad_out
            u_grad_out = _gather_at_dims(grad_out, mapidxs=uao2aos, dims=[-2, -1])

            # get the scatter indices
            ao2shl0 = u_wrappers[0].ao_to_shell()
            ao2shl1 = u_wrappers[1].ao_to_shell()

            # calculate the gradient w.r.t. coeffs
            if allcoeffs.requires_grad:
                grad_allcoeffs = torch.zeros_like(allcoeffs)  # (ngauss)

                # get uncontracted version of integral (..., nu_ao0, nu_ao1)
                dout_dcoeff = _int2c(
                    *u_params,
                    wrappers=u_wrappers,
                    namemgr=int_nmgr,
                )

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)

                # divide done here instead of after scatter to make the 2nd
                # gradient calculation correct. Division can also be done after
                # scatter for more efficient 1st grad calculation, but it gives
                # the wrong result for 2nd grad
                dout_dcoeff_i = dout_dcoeff / coeffs_ao0[:, None]
                dout_dcoeff_j = dout_dcoeff / coeffs_ao1

                # (nu_ao)
                grad_dcoeff_i = einsum("...ij,...ij->i", u_grad_out, dout_dcoeff_i)
                grad_dcoeff_j = einsum("...ij,...ij->j", u_grad_out, dout_dcoeff_j)

                # scatter the grad
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_dcoeff_i)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_dcoeff_j)

            # calculate the gradient w.r.t. alphas
            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                def u_int_fcn(u_wrappers, int_nmgr) -> Tensor:
                    return _int2c(*u_params, wrappers=u_wrappers, namemgr=int_nmgr)

                # get the uncontracted integrals
                sname_derivs = [
                    int_nmgr.get_intgl_deriv_namemgr("rr", ib) for ib in (0, 1)
                ]
                new_axes_pos = [
                    int_nmgr.get_intgl_deriv_newaxispos("rr", ib) for ib in (0, 1)
                ]
                dout_dalphas = _get_integrals(
                    sname_derivs, u_wrappers, u_int_fcn, new_axes_pos
                )

                # (nu_ao)
                # negative because the exponent is negative alpha * (r-ra)^2
                grad_dalpha_i = -einsum("...ij,...ij->i", u_grad_out, dout_dalphas[0])
                grad_dalpha_j = -einsum("...ij,...ij->j", u_grad_out, dout_dalphas[1])
                # grad_dalpha = (grad_dalpha_i + grad_dalpha_j)  # (nu_ao)

                # scatter the grad
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl0, src=grad_dalpha_i)
                grad_allalphas.scatter_add_(dim=-1, index=ao2shl1, src=grad_dalpha_j)

        return (grad_allcoeffs, grad_allalphas, grad_allposs, None, None, None)


class Int2c_V1(BaseInt2c):
    """
    Wrapper class to provide the gradient of the 2-centre integrals.
    """

    @staticmethod
    def forward(
        ctx: CTX,
        allcoeffs: Tensor,
        allalphas: Tensor,
        allposs: Tensor,
        wrappers: list[LibcintWrapper],
        int_nmgr: IntorNameManager,
        hermitian: bool,
    ) -> Tensor:
        # Those tensors are not used directly in the forward calculation, but
        # required for backward propagation:
        # - allcoeffs: (ngauss_tot,)
        # - allalphas: (ngauss_tot,)
        # - allposs: (nat, 3)
        #
        # Wrapper0 and wrapper1 must have the same _atm, _bas, and _env.
        # The check should be done before calling this function.
        assert len(wrappers) == 2

        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.wrappers = wrappers
        ctx.int_nmgr = int_nmgr
        ctx.hermitian = hermitian

        # (..., nao0, nao1)
        return Intor(int_nmgr, wrappers, hermitian=hermitian).calc()


class Int2c_V2(BaseInt2c):
    """
    Wrapper class to provide the gradient of the 2-centre integrals.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(
        allcoeffs: Tensor,
        allalphas: Tensor,
        allposs: Tensor,
        wrappers: list[LibcintWrapper],
        int_nmgr: IntorNameManager,
        hermitian: bool,
    ) -> Tensor:
        # Those tensors are not used directly in the forward calculation, but
        # required for backward propagation:
        # - allcoeffs: (ngauss_tot,)
        # - allalphas: (ngauss_tot,)
        # - allposs: (nat, 3)
        #
        # Wrapper0 and wrapper1 must have the same _atm, _bas, and _env.
        # The check should be done before calling this function.
        assert len(wrappers) == 2

        # (..., nao0, nao1)
        return Intor(int_nmgr, wrappers, hermitian=hermitian).calc()

    @staticmethod
    def setup_context(
        ctx: CTX,
        inputs: tuple[
            Tensor, Tensor, Tensor, list[LibcintWrapper], IntorNameManager, bool
        ],
        output: Tensor,
    ) -> None:
        allcoeffs = inputs[0]
        allalphas = inputs[1]
        allposs = inputs[2]
        wrappers = inputs[3]
        int_nmgr = inputs[4]
        hermitian = inputs[5]

        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.wrappers = wrappers
        ctx.int_nmgr = int_nmgr
        ctx.hermitian = hermitian


################### integrator (direct interface to libcint) ###################


class _CintoptHandler(ctypes.c_void_p):
    """
    Handler for the libcint optimizer.
    """

    def __del__(self):
        try:
            CGTO().CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass


class Intor:
    """
    Interface to the libcint integrals.
    """

    def __init__(
        self,
        int_nmgr: IntorNameManager,
        wrappers: list[LibcintWrapper],
        hermitian: bool = False,
    ) -> None:
        assert len(wrappers) > 0
        wrapper0 = wrappers[0]
        self.int_type = int_nmgr.int_type
        self.atm, self.bas, self.env = wrapper0.atm_bas_env
        self.wrapper0 = wrapper0
        self.int_nmgr = int_nmgr
        self.wrapper_uniqueness = _get_uniqueness([id(w) for w in wrappers])
        self.dd = {"device": wrapper0.device, "dtype": wrapper0.dtype}

        # only use hermitian argument if it is not a derivative
        self.hermitian = hermitian if int_nmgr.order == 0 else False

        # get the operator
        opname = int_nmgr.get_intgl_name(wrapper0.spherical)
        self.op = getattr(CINT(), opname)
        self.optimizer = _get_intgl_optimizer(opname, self.atm, self.bas, self.env)

        # prepare the output
        comp_shape = int_nmgr.get_intgl_components_shape()
        self.outshape = comp_shape + tuple(w.nao() for w in wrappers)
        self.ncomp = reduce(operator.mul, comp_shape, 1)
        self.shls_slice = sum((w.shell_idxs for w in wrappers), ())
        self.integral_done = False

    def calc(self) -> Tensor:
        """
        Calculate the integral.

        Returns
        -------
        Tensor
            Integral tensor.

        Raises
        ------
        ValueError
            If the integral type is unknown.
        """
        assert not self.integral_done
        self.integral_done = True

        if self.int_type in ("int1e", "int2c2e"):
            return self._int2c()

        raise ValueError(f"Unknown integral type: {self.int_type}.")

    def _int2c(self) -> Tensor:
        """
        Calculate the 2-centre integrals with libcint.

        Returns
        -------
        Tensor
            Integral tensor.
        """
        drv = CGTO().GTOint2c
        outshape = self.outshape
        out = np.empty((*outshape[:-2], outshape[-1], outshape[-2]), dtype=np.float64)
        drv(
            self.op,
            out.ctypes.data_as(ctypes.c_void_p),
            int2ctypes(self.ncomp),
            int2ctypes(self.hermitian),
            (ctypes.c_int * len(self.shls_slice))(*self.shls_slice),
            np2ctypes(self.wrapper0.full_shell_to_aoloc),
            self.optimizer,
            np2ctypes(self.atm),
            int2ctypes(self.atm.shape[0]),
            np2ctypes(self.bas),
            int2ctypes(self.bas.shape[0]),
            np2ctypes(self.env),
        )

        out = np.swapaxes(out, -2, -1)
        # TODO: check if we need to do the lines below for 3rd order grad and higher
        # if out.ndim > 2:
        #     out = np.moveaxis(out, -3, 0)
        return numpy_to_tensor(out, **self.dd)


def _get_intgl_optimizer(
    opname: str, atm: np.ndarray, bas: np.ndarray, env: np.ndarray
) -> ctypes.c_void_p:
    """
    Get the optimizer for the integrals.

    Parameters
    ----------
    opname : str
        Name of the integral operator.
    atm : np.ndarray
        Atomic information.
    bas : np.ndarray
        Basis information.
    env : np.ndarray
        Environment information.

    Returns
    -------
    ctypes.c_void_p
        Optimizer for the integrals.
    """
    # setup the optimizer
    cintopt = ctypes.POINTER(ctypes.c_void_p)()
    optname = opname.replace("_cart", "").replace("_sph", "") + "_optimizer"
    copt = getattr(CINT(), optname)
    copt(
        ctypes.byref(cintopt),
        np2ctypes(atm),
        int2ctypes(atm.shape[0]),
        np2ctypes(bas),
        int2ctypes(bas.shape[0]),
        np2ctypes(env),
    )
    opt = ctypes.cast(cintopt, _CintoptHandler)
    return opt


############### name derivation manager functions ###############
def _get_integrals(
    int_nmgrs: list[IntorNameManager],
    wrappers: list[LibcintWrapper],
    int_fcn: Callable[[list[LibcintWrapper], IntorNameManager], Tensor],
    new_axes_pos: list[int | None],
) -> list[Tensor]:
    # Return the list of tensors of the integrals given by the list of integral names.
    # Int_fcn is the integral function that receives the name and returns the results.
    # If new_axes_pos is specified, then move the new axes to 0, otherwise, just leave
    # it as it is

    res: list[Tensor] = []
    # indicating if the integral is available in the libcint-generated file
    int_avail: list[bool] = [False] * len(int_nmgrs)

    for i in range(len(int_nmgrs)):
        res_i: Tensor | None = None

        # check if the integral can be calculated from the previous results
        for j in range(i - 1, -1, -1):
            # check the integral names equivalence
            transpose_path = int_nmgrs[j].get_transpose_path_to(int_nmgrs[i])
            if transpose_path is not None:
                # if the swapped wrappers remain unchanged, then just use the
                # transposed version of the previous version
                # TODO: think more about this (do we need to use different
                # transpose path? e.g. transpose_path[::-1])
                twrappers = _swap_list(wrappers, transpose_path)
                if twrappers == wrappers:
                    res_i = _transpose(res[j], transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # otherwise, use the swapped integral with the swapped wrappers,
                # only if the integral is available in the libcint-generated
                # files
                elif int_avail[j]:
                    res_i = int_fcn(twrappers, int_nmgrs[j])
                    res_i = _transpose(res_i, transpose_path)
                    permute_path = int_nmgrs[j].get_comp_permute_path(transpose_path)
                    res_i = res_i.permute(*permute_path)
                    break

                # if the integral is not available, then continue the searching
                else:
                    continue

        if res_i is None:
            try:
                # successfully executing the line below indicates that the integral
                # is available in the libcint-generated files
                res_i = int_fcn(wrappers, int_nmgrs[i])
            except AttributeError as e:
                msg = f"The integral {int_nmgrs[i].fullname} is not available from libcint, please add it"

                raise AttributeError(msg) from e

            int_avail[i] = True

        res.append(res_i)

    # move the new axes (if any) to dimension 0
    assert res_i is not None
    for i in range(len(res)):
        new_axes_pos_i = new_axes_pos[i]
        if new_axes_pos_i is not None:
            res[i] = torch.movedim(res[i], new_axes_pos_i, 0)

    return res


def _transpose(a: Tensor, axes: list[tuple[int, int]]) -> Tensor:
    # perform the transpose of two axes for tensor a
    for axis2 in axes:
        a = a.transpose(*axis2)
    return a


def _swap_list(a: list, swaps: list[tuple[int, int]]) -> list:
    # swap the elements according to the swaps input
    res = copy.copy(a)  # shallow copy
    for idxs in swaps:
        res[idxs[0]], res[idxs[1]] = res[idxs[1]], res[idxs[0]]  # swap the elements
    return res


def _gather_at_dims(inp: Tensor, mapidxs: list[Tensor], dims: list[int]) -> Tensor:
    # expand inp in the dimension dim by gathering values based on the given
    # mapping indices

    # mapidx: (nnew,) with value from 0 to nold - 1
    # inp: (..., nold, ...)
    # out: (..., nnew, ...)
    out = inp
    for dim, mapidx in zip(dims, mapidxs):
        if dim < 0:
            dim = out.ndim + dim
        map2 = mapidx[(...,) + (None,) * (out.ndim - 1 - dim)]
        map2 = map2.expand(*out.shape[:dim], -1, *out.shape[dim + 1 :])
        out = torch.gather(out, dim=dim, index=map2)
    return out


def _get_uniqueness(a: list) -> list[int]:
    """
    Get the uniqueness pattern from the list.

    Note that this does not sort the list before getting the uniqueness pattern.
    For this reason, PyTorch's `torch.unique` does not yield the same result if
    the first element is not the smallest. PyTorch even ignores the `sorted`
    argument, i.e., `torch.unique(x, return_inverse=True, sorted=False)` is
    equivalent to `torch.unique(x, return_inverse=True, sorted=True)`.
    See https://github.com/pytorch/pytorch/issues/105742.

    Parameters
    ----------
    a : list
        List of elements.

    Returns
    -------
    list[int]
        Uniqueness pattern.

    Examples
    --------
    >>> _get_uniqueness([1, 1, 2, 3, 2])
    [0, 0, 1, 2, 1]
    >>> _get_uniqueness([2, 2, 1, 3, 2])
    [0, 0, 1, 2, 0]
    """
    # Do we need to preserve the order or can we use PyTorch's `torch.unique`,
    # which sorts the result?
    # return torch.unique(torch.tensor(a), return_inverse=True, sorted=False)[1].tolist()
    s: dict = {}
    res: list[int] = []
    i = 0
    for elmt in a:
        if elmt in s:
            res.append(s[elmt])
        else:
            s[elmt] = i
            res.append(i)
            i += 1
    return res
