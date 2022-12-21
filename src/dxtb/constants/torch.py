"""
PyTorch related constants like data types.
"""
from __future__ import annotations

import torch

UINT8: torch.dtype = torch.uint8
"""PyTorch's uint8 data dtype."""

INT16: torch.dtype = torch.int16
"""PyTorch's int16 data dtype."""

INT32: torch.dtype = torch.int32
"""PyTorch's int32 data dtype."""

INT64: torch.dtype = torch.int64
"""PyTorch's int64 data dtype."""

FLOAT16: torch.dtype = torch.float16
"""PyTorch's float16 data dtype."""

FLOAT32: torch.dtype = torch.float32
"""PyTorch's float32 (float) data dtype."""

FLOAT64: torch.dtype = torch.float64
"""PyTorch's float64 (double) data dtype."""
