from __future__ import annotations
import torch

from ..constants import PSE
from ..param import Element
from ..typing import Tensor


def get_pair_param(par_pair: dict[str, float]) -> Tensor:
    symbols = [*PSE.values()]
    pair_mat = torch.ones((len(symbols), len(symbols)))

    for i, isp in enumerate(symbols):
        for j, jsp in enumerate(symbols):
            # Watch format! ("element1-element2")
            pair_mat[i, j] = par_pair.get(
                f"{isp}-{jsp}", par_pair.get(f"{jsp}-{isp}", 1.0)
            )

    return pair_mat


def get_elem_param(par_element: dict[str, Element], key: str) -> Tensor:
    """Obtain a element-wise parametrized quantity for all elements.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).

    Returns
    -------
    Tensor
        Parametrization of all elements (with 0 index being a dummy to allow indexing by atomic numbers).

    Raises
    ------
    ValueError
        If the type of the value of `key` is neither `float` nor `int`.
    """

    # dummy for indexing with atomic numbers
    t = [0.0]

    for item in par_element.values():
        val = getattr(item, key)
        if not isinstance(val, float) and not isinstance(val, int):
            raise ValueError(f"The key '{key}' contains the non-numeric value '{val}'.")

        t.append(val)

    return torch.tensor(t)


def get_elem_param_dict(par_element: dict[str, Element], key: str) -> dict:
    """
    Obtain a element-wise parametrized quantity for all elements.
    Useful for shell-resolved parameters, combines nicely with `IndexHelper`.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).

    Returns
    -------
    dict
        Parametrization of all elements.
    """

    d = {}

    for i, item in enumerate(par_element.values()):
        vals = getattr(item, key)

        # convert shells: [ "1s", "2s" ] -> [ 0, 1 ]
        if isinstance(vals[0], str):
            vals = torch.arange(0, len(vals)).tolist()

        d[i + 1] = vals

    return d


def get_elem_param_shells(par_element: dict[str, Element], key: str = "shells") -> dict:
    """
    Obtain a element-wise parametrized quantity for all elements.
    Useful for shell-resolved parameters, combines nicely with `IndexHelper`.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (default: shells).

    Returns
    -------
    dict
        Parametrization of all elements.
    """

    d = {}

    for i, item in enumerate(par_element.values()):

        # convert shells: [ "1s", "2s" ] -> [ 0, 0 ]
        l = []
        for shell in getattr(item, key):
            l.append({"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}[shell])

        d[i + 1] = l

    return d
