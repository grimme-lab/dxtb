# pylint: disable=protected-access
"""
Test additional utility functions.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.utils import batch, exceptions, is_int_list, is_str_list, set_jit_enabled


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


def test_exceptions() -> None:
    msg = "The error message."
    with pytest.raises(exceptions.SCFConvergenceError, match=msg):
        raise exceptions.SCFConvergenceError(msg)
