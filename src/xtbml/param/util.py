from __future__ import annotations
import torch

from ..constants import PSE
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
    par_element: dict[str, Element],
    key: str,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Obtain a element-wise parametrized quantity for all elements.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.
    key : str
        Name of the quantity to obtain (e.g. gam3 for Hubbard derivatives).
    device : torch.device | None
        Device to store the tensor. If `None` (default), the default device is used.
    dtype : torch.dtype | None
        Data type of the tensor. If `None` (default), the data type is inferred.

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

    return torch.tensor(t, device=device, dtype=dtype)


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

    # print(par_element.get("H"))

    for i, item in enumerate(par_element.values()):
        vals = getattr(item, key)

        if not all((isinstance(x, int) or isinstance(x, float)) for x in vals):
            raise ValueError(
                f"The key '{key}' contains the non-numeric values '{vals}'."
            )

        d[i + 1] = vals

    return d


def get_elem_param_shells(
    par_element: dict[str, Element], valence: bool = False
) -> tuple[dict, dict]:
    """
    Obtain angular momenta of the shells of all atoms.
    This returns the required input for the `IndexHelper`.

    Parameters
    ----------
    par : dict[str, Element]
        Parametrization of elements.

    Returns
    -------
    dict
        Angular momenta of all elements.
    """

    d = {}
    aqm2lsh = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

    if valence:
        v = {}

    for i, item in enumerate(par_element.values()):
        # convert shells: [ "1s", "2s" ] -> [ 0, 0 ]
        l = []
        for shell in getattr(item, "shells"):
            if shell[1] not in aqm2lsh:
                raise ValueError(f"Unknown shell type '{shell[1]}'.")

            l.append(aqm2lsh[shell[1]])

        d[i + 1] = l

        if valence:
            # https://stackoverflow.com/questions/62300404/how-can-i-zero-out-duplicate-values-in-each-row-of-a-pytorch-tensor

            r = torch.tensor(l, dtype=torch.long)
            tmp = torch.ones(r.shape, dtype=torch.bool)
            if r.size(0) < 2:
                v[i + 1] = tmp
                continue

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
            v[i + 1] = torch.gather(tmp, 0, idxs).tolist()

    if valence:
        return d, v

    return d
