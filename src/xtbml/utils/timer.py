"""
Definition of a timer class that can contain multiple timers.
"""

from __future__ import annotations
import time


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


class Timers:
    """Collection of Timers"""

    class Timer:
        """Instance of a Timer"""

        def __init__(self) -> None:
            self._start_time = None
            self.elapsed_time = 0.0

        def start(self) -> None:
            """
            Start a new timer

            Raises
            ------
            TimerError
                If timer is already running.
            """

            if self._start_time is not None:
                raise TimerError("Timer is running. Use .stop() to stop it")

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

            if self._start_time is None:
                raise TimerError("Timer is not running. Use .start() to start it")

            self.elapsed_time += time.perf_counter() - self._start_time
            self._start_time = None

            return self.elapsed_time

    timers: dict[str, Timer]
    """Dictionary of timers"""

    def __init__(self) -> None:
        self.timers = {}

    def start(self, label: str) -> None:
        """
        Create a new timer or start an existing timer named `label`.

        Parameters
        ----------
        label : str
            Name of the timer.
        """

        if label in self.timers:
            t = self.timers[label]
            t.start()
            return

        t = self.Timer()
        t.start()

        self.timers[label] = t

    def stop(self, label: str) -> float:
        """
        Stop the timer

        Parameters
        ----------
        label : str
            Name of the timer.

        Returns
        -------
        float
            Elapsed time in seconds.

        Raises
        ------
        TimerError
            If timer named `label` does not exist.
        """

        if label not in self.timers:
            raise TimerError(f"Timer '{label}' does not exist.")

        t = self.timers[label]
        elapsed_time = t.stop()

        return elapsed_time

    def get_times(self) -> dict[str, float]:
        """
        Get the elapsed times of all timers,

        Returns
        -------
        dict[str, float]
            Dictionary of timer names and elapsed times.
        """

        return {label: t.elapsed_time for label, t in self.timers.items()}

    def print_times(self) -> None:
        """Print the elapsed times of all timers in a table."""
        d = self.get_times()
        total = d.pop("total")
        s = sum(d.values())
        width = 50

        print("{:*^50}".format("Timings"))
        print("")

        print("{:<20} {:<18} {:<12}".format("Objective", "Time in s", "Time in %"))

        print(width * "-")

        for label, t in d.items():
            print("{:<20} {:<18.3f} {:<12.3f}".format(label, t, t / total))

        print(width * "-")
        print("{:<20} {:<18.3f} {:<12.3f}".format("sum", s, s / total))
        print("{:<20} {:<18.3f} {:<12.3f}".format("total", total, total / total))

        print("")
        print(width * "*")
