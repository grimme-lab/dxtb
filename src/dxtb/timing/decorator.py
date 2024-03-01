"""
Definition of a timer class that can contain multiple timers.
"""

from __future__ import annotations

from functools import wraps

from .._types import Any, Callable, TypeVar
from .timer import timer

__all__ = ["timer_decorator"]


F = TypeVar("F", bound=Callable[..., Any])


def timer_decorator(label: str | None = None) -> Callable[[F], F]:
    """
    Decorator for measuring execution time of a function using the global timer.

    Parameters
    ----------
    label : str | None, optional
        Optional label for the timer for more descriptive output.

    Returns
    -------
    Callable[[F], F]
        Decorator function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            uid = func.__name__
            timer.start(uid, label if label is not None else uid)
            result = func(*args, **kwargs)
            timer.stop(uid)

            return result

        return wrapper  # type: ignore

    return decorator
