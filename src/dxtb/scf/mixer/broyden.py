"""
Broyden mixing
==============
"""
from __future__ import annotations

from ..._types import Any
from ...exlibs.xitorch import optimize as xto
from .base import Mixer

default_opts = {
    "method": "broyden1",
    "alpha": -0.5,
    "f_tol": 1.0e-6,
    "x_tol": 1.0e-6,
    "f_rtol": float("inf"),
    "x_rtol": float("inf"),
    "maxiter": 50,
    "verbose": False,
    "line_search": False,
}


class Broyden(Mixer):
    """
    Broyden mixing using xitorch.
    """

    def __init__(self, options: dict[str, Any] | None = None) -> None:
        if options is not None:
            default_opts.update(options)
        super().__init__(default_opts)

    def iter(self):
        raise NotImplementedError
