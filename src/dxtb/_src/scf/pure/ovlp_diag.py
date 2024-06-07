from __future__ import annotations

import torch

from dxtb._src.exlibs import xitorch as xt
from dxtb._src.timing.decorator import timer_decorator
from dxtb._src.typing import Tensor

__all__ = ["get_overlap", "diagonalize"]


def get_overlap(smat: Tensor) -> xt.LinearOperator:
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

    return xt.LinearOperator.m(
        smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)
    )


@timer_decorator("Diagonalize", "SCF")
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

    h_op = xt.LinearOperator.m(hamiltonian)
    o_op = get_overlap(ovlp)

    return xt.linalg.lsymeig(A=h_op, M=o_op, **eigen_options)
