"""
Self-consistent field
=====================
"""

from __future__ import annotations

import torch

from .._types import Tensor
from ..timing.decorator import timer_decorator
from ..utils import eighb
from .base import BaseSCF


class BaseTSCF(BaseSCF):
    """
    Base class for a standard self-consistent field iterator.

    This base class implements the `get_overlap` and the `diagonalize` methods
    using plain tensors. The diagonalization routine is taken from TBMaLT
    (hence the T in the class name).

    This base class only lacks the `scf` method, which implements mixing and
    convergence.
    """

    def get_overlap(self) -> Tensor:
        """
        Get the overlap matrix.

        Returns
        -------
        Tensor
            Overlap matrix.
        """

        smat = self._data.ints.overlap

        zeros = torch.eq(smat, 0)
        mask = torch.all(zeros, dim=-1) & torch.all(zeros, dim=-2)

        return smat + torch.diag_embed(smat.new_ones(*smat.shape[:-2], 1) * mask)

    @timer_decorator("Eigen")
    def diagonalize(self, hamiltonian: Tensor) -> tuple[Tensor, Tensor]:
        """
        Diagonalize the Hamiltonian.

        The overlap matrix is retrieved within this method using the
        `get_overlap` method.

        Parameters
        ----------
        hamiltonian : Tensor
            Current Hamiltonian matrix.

        Returns
        -------
        evals : Tensor
            Eigenvalues of the Hamiltonian.
        evecs : Tensor
            Eigenvectors of the Hamiltonian.
        """
        o = self.get_overlap()

        # FIXME: Not sure if this catches all grad tensors.
        if hamiltonian.requires_grad is False:
            broadening_method = None
        else:
            broadening_method = "lorn"

        return eighb(
            a=hamiltonian,
            b=o,
            is_posdef=True,
            factor=torch.finfo(self.dtype).eps ** 0.5,
            broadening_method=broadening_method,
        )
