"""
Provide eigenvalue solvers with stable backpropagation in case of degeneracies in
the eigenvalue spectrum.
"""

import torch
import functools

from ..typing import Callable, Optional, Tensor, Tuple


def symmetrize(x: Tensor, dim0: int = -1, dim1: int = -2) -> Tensor:
    """
    Symmetrizes the specified tensor

    Parameters
    ----------
    x : Tensor
        The tensor to be symmetrized
    dim0 : int
        First dimension to be transposed
    dim1 : int
        Second dimension to be transposed

    Returns
    -------
    Tensor
        The symmetrized tensor
    """
    return (x + x.transpose(dim0, dim1)) * 0.5


def broadening(func):
    """
    Applies a broadening to the degenerate eigenvalues of a matrix
    """

    class Eigensolver(torch.autograd.Function):
        """
        Solves an eigenvalue problem for a symmetric matrix while applying a conditional
        broadening to the eigenvalues during backpropagation for gradient stablility

        - M. Seeger, A. Hetzel, Z. Dai, and E. Meissner,
          Auto-Differentiating Linear Algebra.
          `ArXiv:1710.08717 <http://arxiv.org/abs/1710.08717>`__
          Aug. 2019.
        """

        @staticmethod
        def forward(
            ctx,
            amat: Tensor,
        ) -> Tuple[Tensor, Tensor]:
            """
            Solves an eigenvalue problem for a symmetric matrix and stores the
            eigenvalues and eigenvectors in the context for backpropagation

            Parameters
            ----------
            amat : Tensor
                The symmetric matrix to be diagonalized
            factor : float
                The factor by which the eigenvalues are broadened

            Returns
            -------
            eigvals : Tensor
                The eigenvalues of the matrix
            eigvecs : Tensor
                The eigenvectors of the matrix
            """

            evals, evecs = func(amat)

            ctx.save_for_backward(evals, evecs)

            return evals, evecs

        @staticmethod
        def backward(ctx, evals_bar, evecs_bar) -> Tensor:
            """
            Backpropagates the eigenvalues and eigenvectors from the forward pass,
            while applying a conditional broadening to the eigenvalues.

            Parameters
            ----------
            evals_bar : Tensor
                The gradient associated to the eigenvalues
            evecs_bar : Tensor
                The gradient associated to the eigenvectors

            Returns
            -------
            amat_bar : Tensor
                The gradient of the loss with respect to the matrix
            """

            # Retrieve eigenvalues and eigenvectors from ctx
            evals, evecs = ctx.saved_tensors
            factor = torch.finfo(evals.dtype).eps ** 0.6
            evecs_t = evecs.transpose(-2, 1)

            # Identify the indices of the upper triangle of the F matrix
            tri_u = torch.triu_indices(*evecs.shape[-2:], 1)

            # Construct the deltas
            deltas = evals[..., tri_u[1]] - evals[..., tri_u[0]]
            deltas = torch.sign(deltas) / torch.where(
                torch.abs(deltas) > factor,
                deltas,
                factor,
            )

            # Construct F matrix where F_ij = evecs_bar_j - evecs_bar_i
            F = evals_bar.new_zeros(*evals.shape, evals.shape[-1])
            F[..., tri_u[0], tri_u[1]] = deltas
            F[..., tri_u[1], tri_u[0]] -= F[..., tri_u[0], tri_u[1]]

            # Contributions from eigenvalues and eigenvectors to gradient
            a_bar = (
                evecs @ (evals_bar.diag_embed() + F * (evecs_t @ evecs_bar)) @ evecs_t
            )

            # Symmetrize to enhance numerical stability
            return symmetrize(a_bar)

    return Eigensolver.apply


def estimate_minmax(
    amat: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Estimate maximum and minimum eigenvalue of a matrix using the Gershgorin circle theorem.

    Parameters
    ----------
    amat : Tensor
        Symmetric matrix.

    Returns
    -------
    Tuple[Tensor, Tensor]
        Minimum and maximum eigenvalue.

    Examples
    --------
    >>> amat = torch.tensor([
    ...     [[-1.1258, -0.1794,  0.1126],
    ...      [-0.1794,  0.5988,  0.1490],
    ...      [ 0.1126,  0.1490,  0.4681]],
    ...     [[-0.1577,  0.6080, -0.3301],
    ...      [ 0.6080,  1.5863,  0.9391],
    ...      [-0.3301,  0.9391,  1.2590]],
    ... ])
    >>> estimate_minmax(amat)
    (tensor([-1.4178, -1.0958]), tensor([0.9272, 3.1334]))
    >>> evals = torch.linalg.eigh(amat)[0]
    >>> evals.min(-1)[0], evals.max(-1)[0],
    (tensor([-1.1543, -0.5760]), tensor([0.7007, 2.4032]))
    """

    center = amat.diagonal(dim1=-2, dim2=-1)
    radius = torch.sum(torch.abs(amat), dim=-1) - torch.abs(center)

    return (
        torch.min(center - radius, dim=-1)[0],
        torch.max(center + radius, dim=-1)[0],
    )


def batched_solve(func: Callable) -> Callable:
    """
    Create batched version of an eigensolver to remove eigenvalues introduced by padding
    """

    @functools.wraps(func)
    def solver(
        amat: Tensor,
        bmat: Optional[Tensor] = None,
        *args,
        cholesky: bool = True,
    ) -> Tuple[Tensor, Tensor]:

        zeros = torch.eq(amat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        if bmat is None:
            shift = estimate_minmax(amat)[-1]
            amat = amat + torch.diag_embed(shift.unsqueeze(-1) * mask)

            evals, evecs = func(amat, *args)

        else:
            bmat = bmat + torch.diag_embed(bmat.new_ones(*bmat.shape[:-2], 1) * mask)

            if cholesky:
                lmat = torch.linalg.solve(
                    torch.linalg.cholesky(bmat),
                    torch.eye(amat.shape[-1], dtype=bmat.dtype, device=bmat.device),
                )
            else:
                svals, svecs = func(bmat, *args)
                lmat = svecs @ torch.diag_embed(svals**-0.5) @ svecs.transpose(-2, -1)

            lmat_t = torch.transpose(lmat, -1, -2)
            cmat = lmat @ amat @ lmat_t

            shift = estimate_minmax(cmat)[-1]
            cmat = cmat + torch.diag_embed(shift.unsqueeze(-1) * mask)

            evals, fvecs = func(cmat, *args)

            evecs = lmat_t @ fvecs

        evals = torch.where(~mask, evals, evals.new_tensor(0))
        return evals, evecs

    return solver


eigh = batched_solve(torch.linalg.eigh)
eighb = batched_solve(broadening(torch.linalg.eigh))
