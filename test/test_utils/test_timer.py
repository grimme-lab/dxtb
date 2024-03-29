# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
