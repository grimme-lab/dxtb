"""
Overlap
=======

The GFNn-xTB overlap matrix.
"""
from __future__ import annotations

import torch

from .._types import Any, Literal, Protocol, Tensor, TensorLike
from ..basis import Basis, IndexHelper
from ..constants import defaults, units
from ..param import Param, get_elem_angular
from ..utils import batch, symmetrize, t2int
from . import libcint as intor
from .mmd import overlap_gto, overlap_gto_grad
from .utils import get_pairs, get_subblock_start

__all__ = ["Overlap", "OverlapLibcint"]


def snorm(ovlp: Tensor) -> Tensor:
    return torch.pow(ovlp.diagonal(dim1=-1, dim2=-2), -0.5)


class OverlapLibcint(TensorLike):
    """
    Overlap integral from atomic orbitals.
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        driver: intor.LibcintWrapper | list[intor.LibcintWrapper] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp
        self.driver = driver

        self._norm = None
        self._matrix = None
        self._gradient = None

    @property
    def norm(self) -> Tensor:
        if self._norm is None:
            raise RuntimeError("Overlap norm has not been calculated.")
        return self._norm

    @norm.setter
    def norm(self, n: Tensor) -> None:
        self._norm = n

    @property
    def matrix(self) -> Tensor:
        if self._matrix is None:
            raise RuntimeError("Overlap matrix has not been calculated.")
        return self._matrix

    @matrix.setter
    def matrix(self, mat: Tensor) -> None:
        self._matrix = mat

    @property
    def gradient(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("Overlap gradient has not been calculated.")
        return self._gradient

    @gradient.setter
    def gradient(self, mat: Tensor) -> None:
        self._gradient = mat

    def build(self, *_: Tensor) -> Tensor:
        """
        Overlap calculation using libcint.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        if self.driver is None:
            raise RuntimeError("Integral driver (libcint interface) missing.")

        def fcn(driver) -> tuple[Tensor, Tensor]:
            s = intor.overlap(driver)
            norm = snorm(s)
            mat = torch.einsum("...ij,...i,...j->...ij", s, norm, norm)
            return mat, norm

        # batched mode
        if self.ihelp.batched:
            assert isinstance(self.driver, list)
            assert isinstance(self.driver, list)

            slist = []
            nlist = []

            for driver in self.driver:
                mat, norm = fcn(driver)
                slist.append(mat)
                nlist.append(norm)

            self.norm = batch.pack(nlist)
            self.matrix = batch.pack(slist)
            return self.matrix

        # single mode
        assert isinstance(self.driver, intor.LibcintWrapper)

        mat, norm = fcn(self.driver)
        self.norm = norm
        self.matrix = mat
        return self.matrix

    def get_gradient(self, *_: Tensor):
        """
        Overlap gradient calculation using libcint.

        Returns
        -------
        Tensor
            Overlap gradient of shape `(nb, norb, norb, 3)`.
        """

        if self.driver is None:
            raise RuntimeError("Integral driver (libcint interface) missing.")

        def fcn(driver: intor.LibcintWrapper, norm: Tensor) -> Tensor:
            # (3, norb, norb)
            grad = intor.int1e("ipovlp", driver)

            # normalize and move xyz dimension to last, which is required for
            # the reduction (only works with extra dimension in last)
            grad = -torch.einsum("...xij,...i,...j->...ijx", grad, norm, norm)
            return grad

        # build norm if not already available
        if self.norm is None:
            if self.ihelp.batched:
                assert isinstance(self.driver, list)
                self.norm = batch.pack(
                    [snorm(intor.overlap(driver)) for driver in self.driver]
                )
            else:
                assert isinstance(self.driver, intor.LibcintWrapper)
                self.norm = snorm(intor.overlap(self.driver))

        # batched mode
        if self.ihelp.batched:
            assert isinstance(self.driver, list)

            glist = []
            for i, driver in enumerate(self.driver):
                norm = batch.deflate(self.norm[i])
                grad = fcn(driver, norm)
                glist.append(grad)

            self.grad = batch.pack(glist)
            return self.grad

        # single mode
        assert isinstance(self.driver, intor.LibcintWrapper)

        self.grad = fcn(self.driver, self.norm)
        return self.grad


class OverlapFunction(Protocol):
    """
    Type annotation for overlap and gradient function.
    """

    def __call__(
        self,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
    ) -> Tensor:
        """
        Evaluation of the overlap integral or its gradient.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        bas : Basis
            Basis set information.
        ihelp : IndexHelper
            Helper class for indexing.
        uplo : Literal['n';, 'u', 'l'], optional
            Whether the matrix of unique shell pairs should be create as a
            triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
            Defaults to `l` (lower triangular matrix).
        cutoff : Tensor | float | int | None, optional
            Real-space cutoff for integral calculation in Angstrom. Defaults to
            `constants.defaults.INTCUTOFF` (50.0).

        Returns
        -------
        Tensor
            Overlap matrix or overlap gradient.
        """
        ...  # pylint: disable=unnecessary-ellipsis


class Overlap(TensorLike):
    """
    Overlap from atomic orbitals.

    Use the `build()` method to calculate the overlap integral. The returned
    matrix uses a custom autograd function to calculate the backward pass with
    the analytical gradient.
    For the full gradient, i.e., a matrix of shape `(nb, norb, norb, 3)`, the
    `get_gradient()` method should be used.
    """

    numbers: Tensor
    """Atomic numbers of the atoms in the system."""

    unique: Tensor
    """Unique species of the system."""

    par: Param
    """Representation of parametrization of xtb model."""

    ihelp: IndexHelper
    """Helper class for indexing."""

    uplo: Literal["n", "u", "l"] = "l"
    """
    Whether the matrix of unique shell pairs should be create as a
    triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
    Defaults to `l` (lower triangular matrix).
    """

    cutoff: Tensor | float | int | None = defaults.INTCUTOFF
    """
    Real-space cutoff for integral calculation in Angstrom. Defaults to
    `constants.defaults.INTCUTOFF` (50.0).
    """

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)
        self.numbers = numbers
        self.unique = torch.unique(numbers)
        self.par = par
        self.ihelp = ihelp
        self.cutoff = cutoff if cutoff is None else cutoff * units.AA2AU

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

    def build(
        self,
        positions: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """
        Overlap calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        mask : Tensor | None
            Mask for positions to make batched computations easier. The overlap
            does not work in a batched fashion. Hence, we loop over the batch
            dimension and must remove the padding. Defaults to `None`, i.e.,
            `batch.deflate()` is used.

        Returns
        -------
        Tensor
            Overlap matrix.
        """
        if self.numbers.ndim > 1:
            s = self._batch(OverlapAG.apply, positions, mask)  # type: ignore
        else:
            s = self._single(OverlapAG.apply, positions)  # type: ignore

        # force symmetry to avoid problems through numerical errors
        if self.uplo == "n":
            return symmetrize(s)
        return s

    def get_gradient(self, positions: Tensor, mask: Tensor | None = None):
        """
        Overlap gradient calculation of unique shells pairs, using the
        McMurchie-Davidson algorithm.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        mask : Tensor | None
            Mask for positions to make batched computations easier. The overlap
            does not work in a batched fashion. Hence, we loop over the batch
            dimension and must remove the padding. Defaults to `None`, i.e.,
            `batch.deflate()` is used.

        Returns
        -------
        Tensor
            Overlap gradient of shape `(nb, norb, norb, 3)`.
        """
        if self.numbers.ndim > 1:
            grad = self._batch(overlap_gradient, positions, mask)
        else:
            grad = self._single(overlap_gradient, positions)

        return grad

    def _single(self, func: OverlapFunction, positions: Tensor) -> Tensor:
        bas = Basis(
            self.unique,
            self.par,
            self.ihelp,
            dtype=self.dtype,
            device=self.device,
        )

        return func(positions, bas, self.ihelp, self.uplo, self.cutoff)

    def _batch(
        self, func: OverlapFunction, positions: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        o = []
        for _batch in range(self.numbers.shape[0]):
            if mask is not None:
                pos = torch.masked_select(
                    positions[_batch],
                    mask[_batch],
                ).reshape((-1, 3))
            else:
                pos = batch.deflate(positions[_batch])

            # unfortunately, we need a new IndexHelper for each batch,
            # but this is much faster than `calc_overlap`
            nums = batch.deflate(self.numbers[_batch])
            ihelp = IndexHelper.from_numbers(nums, get_elem_angular(self.par.element))

            bas = Basis(
                torch.unique(nums),
                self.par,
                ihelp,
                dtype=self.dtype,
                device=self.device,
            )

            o.append(func(pos, bas, ihelp, self.uplo, self.cutoff))

        return batch.pack(o)


class OverlapAG(torch.autograd.Function):
    """
    Autograd function for overlap integral evaluation.
    """

    # TODO: For a more efficient gradient computation, we could also calculate
    # the gradient in the forward pass according to
    # https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html#saving-intermediate-results
    @staticmethod
    def forward(
        ctx: Any,
        positions: Tensor,
        bas: Basis,
        ihelp: IndexHelper,
        uplo: Literal["n", "u", "l"] = "l",
        cutoff: Tensor | float | int | None = defaults.INTCUTOFF,
    ) -> Tensor:
        ctx.save_for_backward(positions)
        ctx.bas = bas
        ctx.ihelp = ihelp
        ctx.uplo = uplo
        ctx.cutoff = cutoff

        return overlap(positions, bas, ihelp, uplo, cutoff)

    @staticmethod
    def backward(
        ctx, grad_out: Tensor
    ) -> tuple[
        None | Tensor,  # positions
        None,  # bas
        None,  # ihelp
        None,  # uplo
        None,  # cutoff
    ]:
        # initialize gradients with `None`
        positions_bar = None

        # check which of the input variables of `forward()` requires gradients
        grad_positions, _, _, _, _ = ctx.needs_input_grad

        positions: Tensor = ctx.saved_tensors[0]
        bas: Basis = ctx.bas
        ihelp: IndexHelper = ctx.ihelp
        uplo: Literal["n", "u", "l"] = ctx.uplo
        cutoff: Tensor | float | int | None = ctx.cutoff

        # analytical gradient for positions
        if grad_positions:
            # We only only calculate the gradient of the overlap w.r.t. one
            # center, but the overlap w.r.t the second center is simply the
            # transpose of the first. The shape of the returned gradient is
            # (nbatch, norb, norb, 3).
            grad_i = overlap_gradient(positions, bas, ihelp, uplo, cutoff)
            grad_j = grad_i.transpose(-3, -2)

            # vjp: (nb, norb, norb) * (nb, norb, norb, 3) -> (nb, norb, 3)
            _gi = torch.einsum("...ij,...ijd->...id", grad_out, grad_i)
            _gj = torch.einsum("...ij,...ijd->...jd", grad_out, grad_j)
            positions_bar = _gi + _gj

            # Finally, we need to reduce the gradient to conform with the
            # expected shape, which is equal to that of the variable w.r.t.
            # which we differentiate, i.e., the positions. Hence, the final
            # reduction does: (nb, norb, 3) -> (nb, natom, 3)
            positions_bar = ihelp.reduce_orbital_to_atom(
                positions_bar, dim=-2, extra=True
            )

        return positions_bar, None, None, None, None


def overlap(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = None,
) -> Tensor:
    """
    Calculate the full overlap matrix.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Angstrom. Defaults to
        `constants.defaults.INTCUTOFF` (50.0).

    Returns
    -------
    Tensor
        Orbital-resolved overlap matrix of shape `(nb, norb, norb)`.
    """
    alphas, coeffs = bas.create_cgtos()

    # spread stuff to orbitals for indexing
    alpha = batch.index(
        batch.index(batch.pack(alphas), ihelp.shells_to_ushell),
        ihelp.orbitals_to_shell,
    )
    coeff = batch.index(
        batch.index(batch.pack(coeffs), ihelp.shells_to_ushell),
        ihelp.orbitals_to_shell,
    )
    pos = batch.index(
        batch.index(positions, ihelp.shells_to_atom),
        ihelp.orbitals_to_shell,
    )
    ang = ihelp.spread_shell_to_orbital(ihelp.angular)

    # real-space integral cutoff; assumes orthogonalization of basis
    # functions as "self-overlap" is  explicitly removed with `dist > 0`
    if cutoff is None:
        mask = None
    else:
        # cdist does not return zero for distance between same vectors
        # https://github.com/pytorch/pytorch/issues/57690
        dist = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
        mask = (dist < cutoff) & (dist > 0.1)

    umap, n_unique_pairs = bas.unique_shell_pairs(mask=mask, uplo=uplo)

    # overlap calculation
    ovlp = torch.zeros(*umap.shape, dtype=positions.dtype, device=positions.device)

    for uval in range(n_unique_pairs):
        pairs = get_pairs(umap, uval)
        first_pair = pairs[0]

        angi, angj = ang[first_pair]
        norbi = 2 * t2int(angi) + 1
        norbj = 2 * t2int(angj) + 1

        # collect [0, 0] entry of each subblock
        upairs = get_subblock_start(umap, uval, norbi, norbj)

        # we only require one pair as all have the same basis function
        alpha_tuple = (
            batch.deflate(alpha[first_pair][0]),
            batch.deflate(alpha[first_pair][1]),
        )
        coeff_tuple = (
            batch.deflate(coeff[first_pair][0]),
            batch.deflate(coeff[first_pair][1]),
        )
        ang_tuple = (angi, angj)

        vec = pos[upairs][:, 0, :] - pos[upairs][:, 1, :]
        stmp = overlap_gto(ang_tuple, alpha_tuple, coeff_tuple, -vec)

        # write overlap of unique pair to correct position in full overlap matrix
        for r, pair in enumerate(upairs):
            ovlp[
                pair[0] : pair[0] + norbi,
                pair[1] : pair[1] + norbj,
            ] = stmp[r]

    # fill empty triangular matrix
    if uplo == "l":
        ovlp = torch.tril(ovlp, diagonal=-1) + torch.triu(ovlp.mT)
    elif uplo == "u":
        ovlp = torch.triu(ovlp, diagonal=1) + torch.tril(ovlp.mT)

    # fix diagonal as "self-overlap" was removed via mask earlier
    ovlp.fill_diagonal_(1.0)
    return ovlp


def overlap_gradient(
    positions: Tensor,
    bas: Basis,
    ihelp: IndexHelper,
    uplo: Literal["n", "u", "l"] = "l",
    cutoff: Tensor | float | int | None = None,
) -> Tensor:
    """
    Calculate the gradient of the overlap.

    Parameters
    ----------
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    bas : Basis
        Basis set information.
    ihelp : IndexHelper
        Helper class for indexing.
    uplo : Literal['n';, 'u', 'l'], optional
        Whether the matrix of unique shell pairs should be create as a
        triangular matrix (`l`: lower, `u`: upper) or full matrix (`n`).
        Defaults to `l` (lower triangular matrix).
    cutoff : Tensor | float | int | None, optional
        Real-space cutoff for integral calculation in Angstrom. Defaults to
        `constants.defaults.INTCUTOFF` (50.0).

    Returns
    -------
    Tensor
        Orbital-resolved overlap gradient of shape `(nb, norb, norb, 3)`.
    """
    alphas, coeffs = bas.create_cgtos()

    # spread stuff to orbitals for indexing
    alpha = batch.index(
        batch.index(batch.pack(alphas), ihelp.shells_to_ushell),
        ihelp.orbitals_to_shell,
    )
    coeff = batch.index(
        batch.index(batch.pack(coeffs), ihelp.shells_to_ushell),
        ihelp.orbitals_to_shell,
    )
    pos = batch.index(
        batch.index(positions, ihelp.shells_to_atom),
        ihelp.orbitals_to_shell,
    )
    ang = ihelp.spread_shell_to_orbital(ihelp.angular)

    # real-space integral cutoff; assumes orthogonalization of basis
    # functions as "self-overlap" is explicitly removed with `dist > 0`
    if cutoff is None:
        mask = None
    else:
        # cdist does not return zero for distance between same vectors
        # https://github.com/pytorch/pytorch/issues/57690
        dist = torch.cdist(pos, pos, compute_mode="donot_use_mm_for_euclid_dist")
        mask = (dist < cutoff) & (dist > 0.1)

    umap, n_unique_pairs = bas.unique_shell_pairs(mask=mask, uplo=uplo)

    # overlap calculation
    ds = torch.zeros((3, *umap.shape), dtype=positions.dtype, device=positions.device)

    # loop over unique pairs
    for uval in range(n_unique_pairs):
        pairs = get_pairs(umap, uval)
        first_pair = pairs[0]

        li, lj = ang[first_pair]
        norbi = 2 * t2int(li) + 1
        norbj = 2 * t2int(lj) + 1

        # collect [0, 0] entry of each subblock
        upairs = get_subblock_start(umap, uval, norbi, norbj, uplo)

        # we only require one pair as all have the same basis function
        alpha_tuple = (
            batch.deflate(alpha[first_pair][0]),
            batch.deflate(alpha[first_pair][1]),
        )
        coeff_tuple = (
            batch.deflate(coeff[first_pair][0]),
            batch.deflate(coeff[first_pair][1]),
        )
        ang_tuple = (li, lj)

        vec = pos[upairs][:, 0, :] - pos[upairs][:, 1, :]

        # NOTE: We could also use the overlap (first return value) here, but
        # the program is structured differently at the moment. Hence, we do not
        # need the overlap here and save us the writing step to the full
        # matrix, which is actually the bottleneck of the overlap routines.
        _, dstmp = overlap_gto_grad(ang_tuple, alpha_tuple, coeff_tuple, -vec)

        # write overlap of unique pair to correct position in full matrix
        for r, pair in enumerate(upairs):
            ds[
                :,
                pair[0] : pair[0] + norbi,
                pair[1] : pair[1] + norbj,
            ] = dstmp[r]

    # fill empty triangular matrix
    if uplo == "l":
        ds = torch.tril(ds, diagonal=-1) - torch.triu(ds.mT)
    elif uplo == "u":
        ds = torch.triu(ds, diagonal=1) - torch.tril(ds.mT)

    # (3, norb, norb) -> (norb, norb, 3)
    return torch.einsum("xij->ijx", ds)
