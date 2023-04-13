"""
Test additional utility functions.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.utils import batch, is_int_list, is_str_list, set_jit_enabled


def test_lists() -> None:
    strlist = ["a", "b"]
    assert is_str_list(strlist)

    intlist = [1, 2]
    assert is_int_list(intlist)

    assert not is_str_list(intlist)
    assert not is_str_list(strlist + intlist)
    assert not is_int_list(strlist)
    assert not is_int_list(strlist + intlist)


def test_jit_settings() -> None:
    # save current state
    state = torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(True)
    assert torch.jit._state._enabled.enabled  # type: ignore

    set_jit_enabled(False)
    assert not torch.jit._state._enabled.enabled  # type: ignore

    # restore initial state
    set_jit_enabled(state)


def test_batch_index() -> None:
    inp = torch.tensor(
        [[0.4800, 0.4701, 0.3405, 0.4701], [0.4701, 0.5833, 0.7882, 0.3542]]
    )
    idx = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3]])
    out = batch.index(inp, idx)
    ref = torch.tensor(
        [
            [0.4800, 0.4800, 0.4701, 0.4701, 0.3405, 0.3405, 0.4701, 0.4701],
            [0.4701, 0.5833, 0.5833, 0.5833, 0.7882, 0.7882, 0.3542, 0.3542],
        ]
    )

    assert pytest.approx(ref) == out

    # different dimensions ( inp.ndim == (idx.ndim + 1) )
    inp = torch.tensor(
        [
            [
                [-3.7510, -5.8131, -1.2251],
                [-1.4523, -3.0188, 2.3872],
                [-1.9942, -3.5295, -1.3030],
                [-4.3375, -6.6594, 0.5598],
            ],
            [
                [3.3579, 2.5251, -3.4608],
                [2.7920, 1.0176, -2.5924],
                [3.0536, 7.1525, 1.8216],
                [1.2930, 0.7893, 0.9190],
            ],
        ]
    )
    idx = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3]])
    out = batch.index(inp, idx)
    ref = torch.tensor(
        [
            [
                [-3.7510, -5.8131, -1.2251],
                [-3.7510, -5.8131, -1.2251],
                [-1.4523, -3.0188, 2.3872],
                [-1.4523, -3.0188, 2.3872],
                [-1.9942, -3.5295, -1.3030],
                [-1.9942, -3.5295, -1.3030],
                [-4.3375, -6.6594, 0.5598],
                [-4.3375, -6.6594, 0.5598],
            ],
            [
                [3.3579, 2.5251, -3.4608],
                [2.7920, 1.0176, -2.5924],
                [2.7920, 1.0176, -2.5924],
                [2.7920, 1.0176, -2.5924],
                [3.0536, 7.1525, 1.8216],
                [3.0536, 7.1525, 1.8216],
                [1.2930, 0.7893, 0.9190],
                [1.2930, 0.7893, 0.9190],
            ],
        ]
    )

    assert pytest.approx(ref) == out

    # different dimensions again ( inp.ndim == (idx.ndim - 1) )
    inp = torch.tensor([0.4800, 0.4701, 0.3405, 0.4701])
    idx = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 1, 2, 2, 3, 3]])
    out = batch.index(inp, idx)
    ref = torch.tensor(
        [
            [0.4800, 0.4800, 0.4701, 0.4701, 0.3405, 0.3405, 0.4701, 0.4701],
            [0.4800, 0.4701, 0.4701, 0.4701, 0.3405, 0.3405, 0.4701, 0.4701],
        ]
    )

    assert pytest.approx(ref) == out
