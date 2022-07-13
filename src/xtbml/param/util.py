from __future__ import annotations
import torch

from ..constants import ATOMIC_NUMBER
from ..param import Element
from ..typing import Tensor, Dict, List


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
    Tensor
        Parametrization of all elements (starting with 1 to allow indexing by atomic numbers).
    """

    d = {}
    for i, item in enumerate(par_element.values()):
        d[i + 1] = getattr(item, key)

    return d


def get_element_angular(par_element: Dict[str, Element]) -> Dict[int, List[int]]:

    label2angular = {
        "s": 0,
        "p": 1,
        "d": 2,
        "f": 3,
        "g": 4,
    }

    return {
        ATOMIC_NUMBER[sym]: [
            label2angular[label[-1]]
            for label in par.shells
        ]
        for sym, par in par_element.items()
    }
