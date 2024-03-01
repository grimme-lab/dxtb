"""
Integral Interface
==================


"""

from __future__ import annotations

import copy
import ctypes
import operator
from functools import reduce

import numpy as np
import torch

try:
    from dxtblibs import CGTO, CINT
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules. {e}. {e.name} provides a Python "
        "interface to the 'libcint' library for fast integral evaluation. "
        "It can be installed via 'pip install dxtblibs'."
    )

from ....._types import Any, Callable, Tensor
from .....utils import einsum
from .namemanager import IntorNameManager
from .utils import int2ctypes, np2ctypes
from .wrapper import LibcintWrapper

__all__ = ["int1e", "overlap"]


def int1e(
    shortname: str,
    wrapper: LibcintWrapper,
    other: LibcintWrapper | None = None,
    hermitian: bool = False,
) -> Tensor:
    # 2-centre 1-electron integral

    # check and set the other parameters
    other1 = _check_and_set(wrapper, other)

    return _Int2cFunction.apply(
        *wrapper.params,
        [wrapper, other1],
        IntorNameManager("int1e", shortname),
        hermitian,
    )  # type: ignore


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


# misc functions
def _check_and_set(
    wrapper: LibcintWrapper, other: LibcintWrapper | None = None
) -> LibcintWrapper:
    # check the value and set the default value of "other" in the integrals
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


############### pytorch functions ###############
class _Int2cFunction(torch.autograd.Function):
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
        # allcoeffs: (ngauss_tot,)
        # allalphas: (ngauss_tot,)
        # allposs: (natom, ndim)
        #
        # Wrapper0 and wrapper1 must have the same _atm, _bas, and _env.
        # The check should be done before calling this function.
        # Those tensors are not used directly in the forward calculation, but
        #   required for backward propagation
        assert len(wrappers) == 2

        out_tensor = Intor(int_nmgr, wrappers, hermitian=hermitian).calc()

        return out_tensor  # (..., nao0, nao1)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Tensor) -> None:
        allcoeffs, allalphas, allposs, wrappers, int_nmgr, hermitian = inputs
        ctx.save_for_backward(allcoeffs, allalphas, allposs)
        ctx.wrappers = wrappers
        ctx.int_nmgr = int_nmgr
        ctx.hermitian = hermitian

    @staticmethod
    def backward(ctx: Any, grad_out: Tensor) -> tuple[Tensor | None, ...]:  # type: ignore
        # grad_out: (..., nao0, nao1)
        allcoeffs: Tensor = ctx.saved_tensors[0]
        allalphas: Tensor = ctx.saved_tensors[1]
        allposs: Tensor = ctx.saved_tensors[2]
        wrappers: list[LibcintWrapper] = ctx.wrappers
        int_nmgr: IntorNameManager = ctx.int_nmgr
        hermitian: bool = ctx.hermitian

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

            def int_fcn(wrappers: list[LibcintWrapper], namemgr) -> Tensor:
                ints = _Int2cFunction.apply(
                    *ctx.saved_tensors, wrappers, namemgr, hermitian
                )
                return ints

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

            # print("\ngrad_dpos_i\n", grad_dpos_i)
            # grad_allpossT is only a view of grad_allposs, so the operation below
            # also changes grad_allposs
            ao_to_atom0 = wrappers[0].ao_to_atom().expand(ndim, -1)
            ao_to_atom1 = wrappers[1].ao_to_atom().expand(ndim, -1)
            # print("ao_to_atom0", ao_to_atom0)
            # print("ao_to_atom1", ao_to_atom1)
            # print("\ngrad_allpossT\n", grad_allpossT)
            # DIMS WRONG
            # h = wrappers[0].ihelp.reduce_orbital_to_atom(grad_dpos_i)
            # print("h", h)

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

                # get the uncontracted version of the integral
                dout_dcoeff = _Int2cFunction.apply(
                    *u_params, u_wrappers, int_nmgr
                )  # (..., nu_ao0, nu_ao1)

                # get the coefficients and spread it on the u_ao-length tensor
                coeffs_ao0 = torch.gather(allcoeffs, dim=-1, index=ao2shl0)  # (nu_ao0)
                coeffs_ao1 = torch.gather(allcoeffs, dim=-1, index=ao2shl1)  # (nu_ao1)
                # divide done here instead of after scatter to make the 2nd gradient
                # calculation correct.
                # division can also be done after scatter for more efficient 1st grad
                # calculation, but it gives the wrong result for 2nd grad
                dout_dcoeff_i = dout_dcoeff / coeffs_ao0[:, None]
                dout_dcoeff_j = dout_dcoeff / coeffs_ao1

                # (nu_ao)
                grad_dcoeff_i = einsum("...ij,...ij->i", u_grad_out, dout_dcoeff_i)
                grad_dcoeff_j = einsum("...ij,...ij->j", u_grad_out, dout_dcoeff_j)
                # grad_dcoeff = grad_dcoeff_i + grad_dcoeff_j

                # scatter the grad
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl0, src=grad_dcoeff_i)
                grad_allcoeffs.scatter_add_(dim=-1, index=ao2shl1, src=grad_dcoeff_j)

            # calculate the gradient w.r.t. alphas
            if allalphas.requires_grad:
                grad_allalphas = torch.zeros_like(allalphas)  # (ngauss)

                def u_int_fcn(u_wrappers, int_nmgr) -> Tensor:
                    return _Int2cFunction.apply(*u_params, u_wrappers, int_nmgr)

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

        return (
            grad_allcoeffs,
            grad_allalphas,
            grad_allposs,
            None,
            None,
            None,
        )


################### integrator (direct interface to libcint) ###################


# Optimizer class
class _cintoptHandler(ctypes.c_void_p):
    def __del__(self):
        try:
            CGTO().CINTdel_optimizer(ctypes.byref(self))
        except AttributeError:
            pass


class Intor:
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
        self.hermitian = hermitian
        self.wrapper_uniqueness = _get_uniqueness([id(w) for w in wrappers])

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
        assert not self.integral_done
        self.integral_done = True

        if self.int_type in ("int1e", "int2c2e"):
            return self._int2c()

        raise ValueError("Unknown integral type: %s" % self.int_type)

    def _int2c(self) -> Tensor:
        # performing 2-centre integrals with libcint
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
        return self._to_tensor(out)

    def _to_tensor(self, out: np.ndarray) -> Tensor:
        # convert the numpy array to the appropriate tensor
        return torch.as_tensor(
            out, dtype=self.wrapper0.dtype, device=self.wrapper0.device
        )


def _get_intgl_optimizer(
    opname: str, atm: np.ndarray, bas: np.ndarray, env: np.ndarray
) -> ctypes.c_void_p:
    # get the optimizer of the integrals
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
    opt = ctypes.cast(cintopt, _cintoptHandler)
    return opt


############### name derivation manager functions ###############
def _get_integrals(
    int_nmgrs: list[IntorNameManager],
    wrappers: list[LibcintWrapper],
    int_fcn: Callable[[list[LibcintWrapper], IntorNameManager], Tensor],
    new_axes_pos: list[int],
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
        if new_axes_pos[i] is not None:
            res[i] = torch.movedim(res[i], new_axes_pos[i], 0)

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
    # get the uniqueness pattern from the list, e.g. _get_uniqueness([1, 1, 2, 3, 2])
    # will return [0, 0, 1, 2, 1]
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
