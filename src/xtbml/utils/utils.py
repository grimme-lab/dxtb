from typing import List
import torch

from xtbml.data.covrad import to_number


def symbol2number(sym_list: List[str]) -> torch.Tensor:
    return torch.flatten(torch.tensor([to_number(s) for s in sym_list]))