"""Halogen bond correction."""

from __future__ import annotations
import torch

from ..data.atomicrad import atomic_rad
from ..param import Param, Element
from ..typing import Tensor
from .base import EnergyContribution


def halogen_bond_correction(
    numbers: Tensor,
    positions: Tensor,
    param: Param,
) -> Tensor:

    if param.halogen is None:
        raise ValueError("No halogen bond correction parameters provided.")

    damp = param.halogen.classical.damping
    rscale = param.halogen.classical.rscale

    rads = atomic_rad[numbers] * rscale
    rcov = rads.unsqueeze(-1) + rads.unsqueeze(-2)

    # strength of halogen bond
    xbond = get_xbond(param.element)
    xbond = xbond[numbers]

    # add epsilon to avoid zero division in some terms
    zero = torch.tensor(0.0, dtype=positions.dtype)
    huge = torch.tensor(torch.finfo(positions.dtype).max, dtype=positions.dtype)

    # masks
    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)

    acceptor_mask = (numbers == 7) | (numbers == 8) | (numbers == 15) | (numbers == 16)
    halogen_mask = (numbers == 17) | (numbers == 35) | (numbers == 53) | (numbers == 85)

    # all distances
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2),
        huge,
    )

    # distance halogen-acceptor
    # TODO: May not work for multiple acceptors
    d_ha = torch.where(halogen_mask, distances[acceptor_mask].squeeze(-2), zero)

    # get closest neighbor of halogen (exluding acceptor)
    min_vals, min_idxs = torch.min(
        torch.where((~acceptor_mask).unsqueeze(-2), distances, huge), dim=-2
    )

    # distance halogen-neighbor
    d_hn = torch.where(
        halogen_mask,
        min_vals,
        zero,
    )

    # distance acceptor-neighbor
    # TODO: batched indexing does not work for single sample
    if len(positions.shape) == 3:
        d_an_t = batched_indexing(distances, min_idxs)
    else:
        d_an_t = distances[min_idxs]

    d_an = torch.transpose(d_an_t, -1, -2)
    d_an = torch.where(halogen_mask, d_an[acceptor_mask].squeeze(-2), zero)

    # Lennard-Jones like potential
    r = torch.where(mask, rcov / distances, zero)
    lj6 = torch.pow(r, 6.0)
    lj12 = torch.pow(lj6, 2.0)
    term = (lj12 - damp * lj6) / (1.0 + lj12)

    # cosine of angle (acceptor-halogen-neighbor) via rule of cosines
    cosa = torch.where(
        halogen_mask, (d_ha * d_ha + d_hn * d_hn - d_an * d_an) / (d_ha * d_hn), zero
    )
    fdamp = torch.pow(0.5 - 0.25 * cosa, 6.0)

    return fdamp * xbond * term[acceptor_mask]


def batched_indexing(inp: Tensor, idx: Tensor) -> Tensor:
    """Batched indexing.

    Parameters
    ----------
    inp : Tensor
        Input tensor.
    idx : Tensor
        Index tensor.

    Returns
    -------
    Tensor
        Output tensor.
    """

    dummy = idx.unsqueeze(-1).expand(idx.size(0), idx.size(-1), inp.size(-1))
    return torch.gather(inp, -2, dummy)


def get_xbond(par_element: dict[str, Element]) -> Tensor:
    """Obtain halogen bond strengths.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    Tensor
        Halogen bond strengths of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
    """

    # dummy for indexing with atomic numbers
    z = [0.0]

    for item in par_element.values():
        z.append(item.xbond)

    return torch.tensor(z)
