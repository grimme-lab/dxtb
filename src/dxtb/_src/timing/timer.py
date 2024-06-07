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
Definition of a timer class that can contain multiple timers.

For developers
--------------
Remember to manually reset the timer in tests that are supposed to fail.
Otherwise, `timer.stop()` may not be called and the next test tries to start
the same timer again, which will throw a (confusing) `TimerError`.
For an example, see `test/test_calculator/test_general.py::test_fail`.
"""

from __future__ import annotations

import time

__all__ = ["timer"]


class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class.
    """


def _sync() -> None:
    """
    Wait for all kernels in all streams on a CUDA device to complete.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _Timers:
    """
    Collection of Timers.
    Upon instantiation, a timer with the label 'total' is started.
    """

    class _Timer:
        """Instance of a Timer."""

        label: str | None
        """Name of the Timer."""

        parent: _Timers
        """Parent Timer collection."""

        _start_time: float | None
        """Time when the timer was started. Should not be accessed directly."""

        elapsed_time: float
        """Elapsed time in seconds."""

        def __init__(
            self, parent: _Timers, label: str | None = None, cuda_sync: bool = False
        ) -> None:
            self.parent = parent
            self.label = label
            self._start_time = None
            self.elapsed_time = 0.0
            self._cuda_sync = cuda_sync

        @property
        def cuda_sync(self) -> bool:
            """
            Check if CUDA synchronization is enabled.

            Returns
            -------
            bool
                Whether CUDA synchronization is enabled (``True``) or not
                (``False``).
            """
            return self._cuda_sync

        @cuda_sync.setter
        def cuda_sync(self, value: bool) -> None:
            """
            Enable or disable CUDA synchronization.

            Parameters
            ----------
            value : bool
                Whether to enable (``True``) or disable (``False``) CUDA
                synchronization.
            """
            self._cuda_sync = value

        def start(self) -> None:
            """
            Start a new timer.

            Raises
            ------
            TimerError
                If timer is already running.
            """
            if not self.parent.enabled:
                return

            if self._start_time is not None:
                raise TimerError(
                    f"Timer '{self.label}' is running. Use `.stop()` to stop it."
                )

            if self.cuda_sync is True:
                _sync()

            self._start_time = time.perf_counter()

        def stop(self) -> float:
            """
            Stop the timer.

            Returns
            -------
            float
                Elapsed time in seconds.

            Raises
            ------
            TimerError
                If timer is not running.
            """
            if not self.parent.enabled:
                return 0.0

            if self._start_time is None:
                raise TimerError(
                    f"Timer '{self.label}' is not running. Use .start() to " "start it."
                )

            if self.cuda_sync is True:
                _sync()

            self.elapsed_time += time.perf_counter() - self._start_time
            self._start_time = None

            return self.elapsed_time

        def is_running(self) -> bool:
            """
            Check if the timer is running.

            Returns
            -------
            bool
                Whether the timer currently runs (``True``) or not (``False``).
            """
            if self._start_time is not None and self.elapsed_time == 0.0:
                return True
            return False

    timers: dict[str, _Timer]
    """Dictionary of timers."""

    label: str | None
    """Name for the Timer collection."""

    def __init__(
        self,
        label: str | None = None,
        autostart: bool = False,
        cuda_sync: bool = False,
        only_parents: bool = False,
    ) -> None:
        self.label = label
        self.timers = {}
        self._enabled = True
        self._subtimer_parent_map = {}
        self._autostart = autostart
        self._cuda_sync = cuda_sync
        self._only_parents = only_parents

        if self._autostart is True:
            self.reset()

    def enable(self) -> None:
        """
        Enable all timers in the collection.
        """
        self._enabled = True

    def disable(self) -> None:
        """
        Disable and reset all timers in the collection.
        """
        self._enabled = False

    @property
    def enabled(self) -> bool:
        """
        Check if the timer is enabled.

        Returns
        -------
        bool
            Whether the timer is enabled (``True``) or not (``False``).
        """
        return self._enabled

    @property
    def cuda_sync(self) -> bool:
        """
        Check if CUDA synchronization is enabled.

        Returns
        -------
        bool
            Whether CUDA synchronization is enabled (``True``) or not
            (``False``).
        """
        return self._cuda_sync

    @cuda_sync.setter
    def cuda_sync(self, value: bool) -> None:
        """
        Enable or disable CUDA synchronization.

        Parameters
        ----------
        value : bool
            Whether to enable (``True``) or disable (``False``) CUDA
            synchronization.
        """
        self._cuda_sync = value

        for t in self.timers.values():
            t.cuda_sync = value

    @property
    def only_parents(self) -> bool:
        """
        Check if only parent timers are enabled.

        Returns
        -------
        bool
            Whether only parent timers are enabled (``True``) or not
            (``False``).
        """
        return self._only_parents

    @only_parents.setter
    def only_parents(self, value: bool) -> None:
        """
        Enable or disable only parent timers.

        Parameters
        ----------
        value : bool
            Whether to enable (``True``) or disable (``False``) only parent
            timers.
        """
        self._only_parents = value

    def start(
        self, uid: str, label: str | None = None, parent_uid: str | None = None
    ) -> None:
        """
        Create a new timer or start an existing timer with `uid`.

        Parameters
        ----------
        uid : str
            ID of the timer.
        label : str | None
            Name of the timer (used for printing). Defaults to ``None``.
            If no `label` is given, the `uid` is used.
        """
        if not self._enabled:
            return

        if self.only_parents is True and parent_uid is not None:
            return

        if uid in self.timers:
            self.timers[uid].start()
            return

        t = self._Timer(self, uid if label is None else label, self.cuda_sync)
        t.start()

        self.timers[uid] = t

        if parent_uid is not None:
            if parent_uid in self.timers:
                self._subtimer_parent_map[uid] = parent_uid

    def stop(self, uid: str) -> float:
        """
        Stop the timer

        Parameters
        ----------
        uid : str
            Unique ID of the timer.

        Returns
        -------
        float
            Elapsed time in seconds.

        Raises
        ------
        TimerError
            If timer dubbed `uid` does not exist.
        """
        if not self.enabled:
            return 0.0

        if uid not in self.timers:
            # If sub timers are disabled, some timers will not exist. So,
            # instead of raising an error, we return just 0.0.
            if self.only_parents is True:
                return 0.0

            raise TimerError(f"Timer '{uid}' does not exist.")

        t = self.timers[uid]
        elapsed_time = t.stop()

        return elapsed_time

    def stop_all(self) -> None:
        """Stop all running timers."""
        for t in self.timers.values():
            if t.is_running():
                t.stop()

    def reset(self) -> None:
        """
        Reset all timers in the collection.

        This method reinitializes the timers dictionary and restarts the
        'total' timer.
        """
        self.timers = {}
        self.start("total")

    def kill(self) -> None:
        """
        Disable, reset and stop all timers.
        """
        self.disable()
        self.reset()
        self.stop_all()

    def get_time(self, uid: str) -> float:
        """
        Get the elapsed time of a timer.

        Parameters
        ----------
        uid : str
            Unique ID of the timer.

        Returns
        -------
        float
            Elapsed time in seconds.

        Raises
        ------
        TimerError
            If timer dubbed `uid` does not exist.
        """
        if not self.enabled:
            return 0.0

        if uid not in self.timers:
            raise TimerError(f"Timer '{uid}' does not exist.")

        return self.timers[uid].elapsed_time

    def get_times(self) -> dict[str, dict[str, float]]:
        """
        Get the elapsed times of all timers,

        Returns
        -------
        dict[str, float]
            Dictionary of timer IDs and elapsed times.
        """
        if self.timers["total"].is_running():
            self.timers["total"].stop()

        KEY = "value"
        times = {}

        # Initialize all parent timers in the times dictionary
        for k in self.timers:
            if k not in self._subtimer_parent_map:
                times[k] = {KEY: None, "sub": {}}

        # Add times for all timers, categorizing based on the parent map
        for uid, t in self.timers.items():
            if uid in self._subtimer_parent_map:
                parent = self._subtimer_parent_map[uid]
                times[parent]["sub"][uid] = t.elapsed_time
            else:
                times[uid][KEY] = t.elapsed_time

        total_time = times["total"][KEY]
        for main_timer, details in times.items():
            if main_timer == "total":
                continue

            # Calculate the percentage of the total time for main timers
            main_time = details[KEY]
            percentage_of_total = (main_time / total_time) * 100
            details["percentage"] = f"{percentage_of_total:.2f}"

            # Calculate the percentage relative to the parent timer for sub
            if details["sub"]:
                for subtimer, sub_time in details["sub"].items():
                    percentage_of_parent = (sub_time / main_time) * 100
                    details["sub"][subtimer] = {
                        KEY: sub_time,
                        "percentage": f"{percentage_of_parent:.2f}",
                    }

        return times

    def print(self, v: int = 5, precision: int = 3) -> None:  # pragma: no cover
        """Print the elapsed times of all timers in a table."""
        if not self._enabled:
            return

        if self.timers["total"].is_running():
            self.timers["total"].stop()

        # pylint: disable=import-outside-toplevel
        from ..io import OutputHandler

        OutputHandler.write_table(
            self.get_times(),
            title="Timings",
            columns=["Objective", "Time (s)", "% Total"],
            v=v,
            precision=precision,
        )


timer = _Timers(autostart=True, cuda_sync=False)
"""Global instance of the timer class."""
