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


