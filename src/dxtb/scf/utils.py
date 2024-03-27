from __future__ import annotations

from .._types import Tensor
from ..timing.decorator import timer_decorator
from ..utils import einsum

__all__ = ["get_density"]


@timer_decorator("Density", "SCF")
def get_density(coeffs: Tensor, occ: Tensor, emo: Tensor | None = None) -> Tensor:
    """
    Calculate the density matrix from the coefficient vector and the occupation.

    Parameters
    ----------
    evecs : Tensor
        MO coefficients.
    occ : Tensor
        Occupation numbers (diagonal matrix).
    emo : Tensor | None, optional
        Orbital energies for energy weighted density matrix. Defaults to `None`.

    Returns
    -------
    Tensor
        (Energy-weighted) Density matrix.
    """
    o = occ if emo is None else occ * emo

    # equivalent: coeffs * o.unsqueeze(-2) @ coeffs.mT
    return einsum("...ik,...k,...jk->...ij", coeffs, o, coeffs)
