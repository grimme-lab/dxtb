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

        self.start("total")

    def enable(self) -> None:
        """
        Enable all timers in the collection.
        """
        self._enabled = True

    def disable(self) -> None:
        """
        Disable all timers in the collection.
        """
        self._enabled = False

    def start(self, uid: str, label: str | None = None) -> None:
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
            t = self.timers[uid]
            t.start()
            return

        t = self._Timer(self, uid if label is None else label)
        t.start()

        self.timers[uid] = t

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

    def get_times(self) -> dict[str, float]:
        """
        Get the elapsed times of all timers,

        Returns
        -------
        dict[str, float]
            Dictionary of timer IDs and elapsed times.
        """
        return {
            (uid if t.label is None else t.label): t.elapsed_time
            for uid, t in self.timers.items()
        }

    def print_times(
        self, name: str = "Timings", width: int = 55
    ) -> None:  # pragma: no cover
        """Print the elapsed times of all timers in a table."""
        if not self._enabled:
            return

        if self.timers["total"].is_running():
            self.timers["total"].stop()

        d = self.get_times()
        total = d.pop("total")
        s = sum(d.values())

        w1, w2, w3 = 25, 18, 12

        print(f"{name:*^55}\n")
        print(f"{'Objective':<{w1}} {'Time in s':<{w2}} {'Time in %':<{w3}}")
        print(width * "-")

        for uid, t in d.items():
            perc = t / total * 100
            print(f"{uid:<25} {t:<{w2}.3f} {perc:^10.2f}")

        print(width * "-")
        print(f"{'sum':<{w1}} {s:<{w2}.3f} {s / total * 100:^9.2f}")
        print(f"{'total':<{w1}} {total:<{w2}.3f} {total / total * 100:^9.2f}")

        print("")


timer = _Timers()
"""Global instance of the timer class."""
