"""
Utility
=======

Collection of utility functions.
"""

from __future__ import annotations
from functools import wraps
from time import time
import torch

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
