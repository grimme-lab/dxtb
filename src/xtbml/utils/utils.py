"""
Utility
=======

Collection of utility functions.
"""

from __future__ import annotations
from functools import wraps
from time import time
from typing import Any
import torch
import functools

from ..constants import ATOMIC_NUMBER
from ..typing import Tensor


def symbol2number(sym_list: list[str]) -> Tensor:
    return torch.flatten(torch.tensor([ATOMIC_NUMBER[s.title()] for s in sym_list]))


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        ts = time()
        result = f(*args, **kwargs)
        te = time()
        print(f"func '{f.__name__}' took: {te-ts:2.4f} sec")
        return result

    return wrap


def dict_reorder(d: dict) -> dict:
    """Reorder a dictionary by keys. Includes sorting of sub-directories.
    Courtesy to https://stackoverflow.com/questions/9001509/how-can-i-sort-a-dictionary-by-key/47017849#47017849


    Parameters
    ----------
    d : dict
        Dictionary to be sorted

    Returns
    -------
    dict
        Sorted dictionary
    """
    return {
        k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(d.items())
    }


@torch.jit.script
def real_atoms(numbers: Tensor) -> Tensor:
    return numbers != 0


@torch.jit.script
def real_pairs(numbers: Tensor, diagonal: bool = False) -> Tensor:
    real = real_atoms(numbers)
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    if not diagonal:
        mask *= ~torch.diag_embed(torch.ones_like(real))
    return mask


###  Helper functions for accesing nested object properties. ###
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rdelattr(obj, attr):
    pre, _, post = attr.rpartition(".")
    return delattr(rgetattr(obj, pre) if pre else obj, post)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def get_attribute_name_key(name: str) -> tuple[str, str]:
    """Get name of nested attributes including dict key for given input.

    Parameters
    ----------
    name : str
        Input string containing name of object attribute (nested and possibly containing dict entry).

    Returns
    -------
    tuple[str, str]
        The name of nested attribute and if present the key of dictionary entry.

    Raises
    ------
    AttributeError
        If input string contains more than single '[', e.g. multiple dictionary entries

    Example:
    > input = "hamiltonian.xtb.kpair['Pt-Ti']"
    > print(get_attribute_name_key(input))
    ('hamiltonian.xtb.kpair', 'Pt-Ti')
    """
    key = None
    split = name.split("[")
    if len(split) > 1:
        name, key = split
        for s in ["'", '"', "]"]:
            key = key.replace(s, "")
    elif len(split) > 2:
        raise AttributeError
    return name, key


def get_all_entries(obj: Any, name: str) -> str | list[str]:
    """Get all entries from dict-like object attribute

    Parameters
    ----------
    obj : Any
        Object to be parsed for attribute
    name : str
        Nested attribute identifier

    Returns
    -------
    str | list[str]
        Name of the attribute or names of attributes for all keys in dict.
    """
    attr = rgetattr(obj, name)

    if isinstance(attr, dict):
        # create string for every key in dict
        return [f"{name}['{k}']" for k in attr]
    else:
        return name
