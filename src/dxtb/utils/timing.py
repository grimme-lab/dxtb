"""
Decorator for timing.
"""
from __future__ import annotations

import time
from functools import wraps
from typing import cast

from .._types import Any, Callable, TypeVar

__all__ = ["timings"]


F = TypeVar("F", bound=Callable[..., Any])


def timings(repeats: int = 1) -> Callable[[F], F]:  # pragma: no cover
    """
    Decorator that prints execution time of a function.

    Parameters
    ----------
    f : Callable
        Function for which execution time should be timed.

    Returns
    -------
    Any
        Return value of input function.
    """

    def decorator(f: F) -> F:
        @wraps(f)
        def wrap(*args: Any, **kwargs: Any) -> Any:
            if repeats < 1:
                return f(*args, **kwargs)

            times = []
            for _ in range(repeats):
                ts = time.perf_counter()
                result = f(*args, **kwargs)
                te = time.perf_counter()
                times.append(te - ts)

                if repeats == 1:
                    print(f"{f.__name__}: {te - ts:2.4f}", end=" ")
                else:
                    print(f"{te - ts:2.4f}", end=" ")

            print()
            if repeats > 1:
                avg_time = sum(times) / len(times)
                print(f"Average ({repeats}) of {f.__name__}: {avg_time:2.4f} sec")

            return result  # type: ignore

        return cast(F, wrap)

    return decorator
