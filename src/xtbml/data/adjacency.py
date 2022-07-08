import torch
from typing import Callable

from .samples import Sample
from ..ncoord import get_coordination_number, exp_count
from ..bond import guess_bond_order
from ..typing import Tensor


def calc_adj(
    sample: Sample,
    cn_fn: Callable[[Tensor, Tensor, float], Tensor] = exp_count,
    masking: bool = True,
) -> Tensor:
    """Calculate the adjacency matrix for a given sample. Depending on the given counting function, bond orders are inferred.
        If masking is turned off, bond orders are returned. Due to (normally) padding of dataset this should have a constant
        shape (i.e. [bs, n_atms, n_atms]).

    Args:
        sample (Sample): Calculate adjacency matrix for given sample topology.
        cn_fn (Callable[[Tensor, Tensor, float], Tensor], optional): Counting function for estimation of bond orders. Defaults to exp_count.
        masking (bool, optional): Masking of bond orders to adjacency matrix. Defaults to True.

    Returns:
        Tensor: Adjacency matrix of given sample (or bond order if masking set False).
    """

    cn = get_coordination_number(
        sample.numbers.type(torch.int64), sample.positions, cn_fn
    )
    bond_order = guess_bond_order(
        sample.numbers.type(torch.int64), sample.positions, cn
    )

    if masking:
        # apply boolean masking
        threshold = 0.0
        adj = torch.where(bond_order > threshold, 1, 0)
        return adj
    else:
        return bond_order
