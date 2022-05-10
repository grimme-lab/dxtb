from typing import List
from ..typing import Tensor
import torch

from ..data.covrad import to_number


def symbol2number(sym_list: List[str]) -> Tensor:
    return torch.flatten(torch.tensor([to_number(s) for s in sym_list]))
