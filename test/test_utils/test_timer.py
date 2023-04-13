"""
Test the `Timer` and Timer collections (`Timers`).
"""
from __future__ import annotations

import pytest

from dxtb.utils import TimerError, Timers


def test_fail() -> None:
    timer = Timers()

    # try to stop a timer that was never started
    with pytest.raises(TimerError):
        timer.stop("test")

    # try to start a timer that is already running
    timer.start("test")
    with pytest.raises(TimerError):
        timer.start("test")

    # stop the timer and try to stop it again
    timer.stop("test")
    with pytest.raises(TimerError):
        timer.timers["test"].stop()
