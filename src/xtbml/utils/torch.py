from __future__ import annotations

import torch

from ..typing import Tensor


def maybe_move(x: Tensor, device: torch.device, dtype: torch.dtype) -> Tensor:
    if x.device != device:
        x = x.to(device)
    if x.dtype != dtype:
        x = x.type(dtype)
    return x
