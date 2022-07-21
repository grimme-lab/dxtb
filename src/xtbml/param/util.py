"""
Parametrization Utility
=======================

Contains functions to obtain the parametrization of elements and pairs.
Most functions convert the parametrization dictionary to a tensor. 
"""

from __future__ import annotations
import torch

from ..constants import ATOMIC_NUMBER, PSE
from ..param import Element
from ..typing import Tensor


def get_pair_param(
    symbols: list[str | int],
    par_pair: dict[str, float],
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Obtain tensor of a pair-wise parametrized quantity for all pairs.

    Parameters
    ----------
    symbols : list[str | int]
        List of atomic symbols or atomic numbers.
    par_pair : dict[str, float]
        Parametrization of pairs.
    device : torch.device | None, optional
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype | None, optional
        Data type of the tensor. If `None` (default), the data type is inferred.

    Returns
    -------
    Tensor
        Parametrization of all pairs of `symbols`.
    """

    # convert atomic numbers to symbols
    if all(isinstance(x, int) for x in symbols):
        symbols = [PSE.get(i, "X") for i in symbols]

    pair_mat = torch.ones(len(symbols), len(symbols), device=device, dtype=dtype)
    for i, isp in enumerate(symbols):
        for j, jsp in enumerate(symbols):
            # Watch format! ("element1-element2")
            pair_mat[i, j] = par_pair.get(
                f"{isp}-{jsp}", par_pair.get(f"{jsp}-{isp}", 1.0)
            )

    return pair_mat


def get_elem_param(
    numbers: Tensor,
    par_element: dict[str, Element],
    key: str,
    pad_val: float = -1.0,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Obtain a element-wise parametrized quantity for selected atomic numbers.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers.
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).
    pad_val : float, optional
        Value to pad the tensor with. Default is `-1.0`.
    device : torch.device | None
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype | None
        Data type of the tensor. If `None` (default), the data type is inferred.

    Returns
    -------
    Tensor
        Parametrization of selected elements.

    Raises
    ------
    ValueError
        If the type of the value of `key` is neither `float` nor `int`.
    """
    l = []

    for number in numbers:
        el = PSE.get(int(number.item()), "X")
        if el in par_element:
            vals = getattr(par_element[el], key)

            # convert to list so that we can use the same function
            # for atom-resolved parameters too
            if isinstance(vals, float):
                vals = [vals]

            if not all(isinstance(x, (int, float)) for x in vals):
                raise ValueError(
                    f"The key '{key}' contains the non-numeric values '{vals}'."
                )

        else:
            vals = [pad_val]

        for val in vals:
            l.append(val)

    return torch.tensor(l, device=device, dtype=dtype)


def get_element_angular(par_element: dict[str, Element]) -> dict[int, list[int]]:
    """Obtain angular momenta of the shells of all atoms.

    Parameters
    ----------
    par_element : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    dict[int, list[int]]
        Angular momenta of all elements.
    """

    label2angular = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
    }

    return {
        ATOMIC_NUMBER[sym]: [label2angular[label[-1]] for label in par.shells]
        for sym, par in par_element.items()
    }


def get_elem_valence(
    numbers: Tensor,
    par_element: dict[str, Element],
    pad_val: int = -1,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Obtain valence of the shells of all atoms.

    Parameters
    ----------
    par_element : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    dict[int, list[int]]
        Valence of all elements.
    """

    l = []
    key = "shells"
    label2angular = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
    }

    for number in numbers:
        el = PSE.get(int(number.item()), "X")
        shells = []
        if el in par_element:
            for shell in getattr(par_element[el], key):
                shell = shell[-1]
                if shell not in label2angular:
                    raise ValueError(f"Unknown shell type '{shell}'.")

                shells.append(label2angular[shell])

        else:
            shells = [pad_val]

        # https://stackoverflow.com/questions/62300404/how-can-i-zero-out-duplicate-values-in-each-row-of-a-pytorch-tensor
        r = torch.tensor(shells, dtype=torch.long)
        tmp = torch.ones(r.shape, dtype=torch.bool)
        if r.size(0) < 2:
            vals = tmp
        else:
            # sorting the rows so that duplicate values appear together
            # e.g. [1, 2, 3, 3, 3, 4, 4]
            y, idxs = torch.sort(r)

            # subtracting, so duplicate values will become 0
            # e.g. [1, 2, 3, 0, 0, 4, 0]
            tmp[1:] = (y[1:] - y[:-1]) != 0

            # retrieving the original indices of elements
            _, idxs = torch.sort(idxs)

            # re-organizing the rows following original order
            # e.g. [1, 2, 3, 4, 0, 0, 0]
            vals = torch.gather(tmp, 0, idxs).tolist()

        for val in vals:
            l.append(val)

    return torch.tensor(l, device=device, dtype=dtype)
