from __future__ import annotations
import torch

from ..param import Element
from ..typing import Tensor


def get_element_param(par_element: dict[str, Element], key: str) -> Tensor:
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
    """

    # dummy for indexing with atomic numbers
    t = [0.0]

    for item in par_element.values():
        t.append(getattr(item, key))

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
    Tensor
        Parametrization of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
    """

    d = {}

    for i, item in enumerate(par_element.values()):
        d[i + 1] = getattr(item, key)

    return d