"""
Typing for the overlap functions.
"""

from __future__ import annotations

from ....._types import Literal, Protocol, Tensor
from .....basis import Basis, IndexHelper
from .....constants import defaults

__all__ = ["OverlapFunction"]


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
            Cartesian coordinates of all atoms in the system (nat, 3).
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
