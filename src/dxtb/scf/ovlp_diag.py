from __future__ import annotations

import torch

from .._types import Tensor
from ..exlibs.xitorch import LinearOperator
from ..exlibs.xitorch import linalg as xtl


def get_overlap(smat: Tensor) -> LinearOperator:
    """
    Get the overlap matrix.

    Parameters
    ----------
    smat : Tensor
        Current overlap matrix as tensor.

    Returns
    -------
    LinearOperator
        Overlap matrix as linear operator.
    """

    zeros = torch.eq(smat, 0)
    mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

    return LinearOperator.m(
        smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
    )


def diagonalize(
    hamiltonian: Tensor, ovlp: Tensor, eigen_options: dict
) -> tuple[Tensor, Tensor]:
    """
    Diagonalize the Hamiltonian.

    Parameters
    ----------
    hamiltonian : Tensor
        Current Hamiltonian matrix.
    ovlp : Tensor
        Current overlap matrix.
    eigen_options : dict
        Options for calculating EVs.

    Returns
    -------
    evals : Tensor
        Eigenvalues of the Hamiltonian.
    evecs : Tensor
        Eigenvectors of the Hamiltonian.
    """

    h_op = LinearOperator.m(hamiltonian)
    o_op = get_overlap(ovlp)

    return xtl.lsymeig(A=h_op, M=o_op, **eigen_options)
