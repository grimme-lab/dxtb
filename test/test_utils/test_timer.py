"""
Test the `Timer` and Timer collections (`Timers`).
"""

from __future__ import annotations

import pytest

from dxtb.timing.timer import TimerError, _Timers


def test_fail() -> None:
    timer = _Timers()

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


def test_running() -> None:
    timer = _Timers()
    timer.start("test")
    assert timer.timers["test"].is_running()

    timer.stop("test")
    assert not timer.timers["test"].is_running()


def test_stopall() -> None:
    timer = _Timers()
    timer.start("test")
    timer.start("test2")
    timer.start("test3")

    timer.stop("test")
    timer.stop_all()

    assert not timer.timers["test"].is_running()
    assert not timer.timers["test2"].is_running()
