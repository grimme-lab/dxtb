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
"""

from __future__ import annotations

import time
from functools import wraps

from dxtb._src.typing import Any, Callable, TypeVar, cast

from .timer import timer

__all__ = ["timer_decorator", "timer_decorator_print"]


F = TypeVar("F", bound=Callable[..., Any])


def timer_decorator(
    label: str | None = None, parent_uid: str | None = None
) -> Callable[[F], F]:
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
            uid = label if label is not None else func.__name__
            timer.start(uid, parent_uid=parent_uid)
            result = func(*args, **kwargs)
            timer.stop(uid)

            return result

        return wrapper  # type: ignore

    return decorator


def timer_decorator_print(repeats: int = 1) -> Callable[[F], F]:  # pragma: no cover
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
