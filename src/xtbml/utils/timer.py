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

        def start(self):
            """Start a new timer"""

            if self._start_time is not None:
                raise TimerError("Timer is running. Use .stop() to stop it")

            self._start_time = time.perf_counter()

        def stop(self) -> float:
            """Stop the timer, and report the elapsed time"""

            if self._start_time is None:
                raise TimerError("Timer is not running. Use .start() to start it")

            self.elapsed_time = time.perf_counter() - self._start_time
            self._start_time = None
            return self.elapsed_time

    timers: dict[str, Timer]
    """Dictionary of timers"""

    def __init__(self) -> None:
        self.timers = {}

    def start(self, label) -> None:
        """Start a new timer"""

        t = self.Timer()
        t.start()

        if label in self.timers:
            raise TimerError(f"Timer '{label}' does already exist and is running.")

        self.timers[label] = t

    def stop(self, label) -> float:
        """Stop the timer, and report the elapsed time"""

        if label not in self.timers:
            raise TimerError(f"Timer '{label}' does not exist.")

        t = self.timers[label]
        elapsed_time = t.stop()

        return elapsed_time

    def get_times(self) -> dict[str, float]:
        """Get the elapsed times of all timers"""

        return {label: t.elapsed_time for label, t in self.timers.items()}

    def print_times(self) -> None:
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
