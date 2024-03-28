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


class _Timers:
    """
    Collection of Timers.
    Upon instantiation, a timer with the label 'total' is started.
    """

    class _Timer:
        """Instance of a Timer."""

        label: str | None
        """Name of the Timer."""

        _start_time: float | None

        def __init__(self, parent: _Timers, label: str | None = None) -> None:
            self.parent = parent
            self.label = label
            self._start_time = None
            self.elapsed_time = 0.0

        def start(self) -> None:
            """
            Start a new timer.

            Raises
            ------
            TimerError
                If timer is already running.
            """
            if not self.parent._enabled:
                return

            if self._start_time is not None:
                raise TimerError(
                    f"Timer '{self.label}' is running. Use `.stop()` to stop it."
                )

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
            if not self.parent._enabled:
                return 0.0

            if self._start_time is None:
                raise TimerError(
                    f"Timer '{self.label}' is not running. Use .start() to " "start it."
                )

            self.elapsed_time += time.perf_counter() - self._start_time
            self._start_time = None

            return self.elapsed_time

        def is_running(self) -> bool:
            """
            Check if the timer is running.

            Returns
            -------
            bool
                Whether the timer currently runs (`True`) or not (`False`).
            """
            if self._start_time is not None and self.elapsed_time == 0.0:
                return True
            return False

    timers: dict[str, _Timer]
    """Dictionary of timers."""

    label: str | None
    """Name for the Timer collection."""

    def __init__(self, label: str | None = None) -> None:
        self.label = label
        self.timers = {}
        self._enabled = True
        self._subtimer_parent_map = {}

        self.start("total")

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
            Name of the timer (used for printing). Defaults to `None`.
            If no `label` is given, the `uid` is used.
        """
        if not self._enabled:
            return

        if uid in self.timers:
            self.timers[uid].start()
            return

        t = self._Timer(self, uid if label is None else label)
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
        if not self._enabled:
            return 0.0

        if uid not in self.timers:
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
        for k in self.timers.keys():
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
            times[main_timer]["percentage"] = f"{percentage_of_total:.2f}"

            # Calculate the percentage relative to the parent timer for sub
            if details["sub"]:
                for subtimer, sub_time in details["sub"].items():
                    percentage_of_parent = (sub_time / main_time) * 100
                    times[main_timer]["sub"][subtimer] = {
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


timer = _Timers()
"""Global instance of the timer class."""
