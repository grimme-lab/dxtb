from typing import List
import torch

from ..data.covrad import to_number


def symbol2number(sym_list: List[str]) -> torch.Tensor:
    return torch.flatten(torch.tensor([to_number(s) for s in sym_list]))


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
