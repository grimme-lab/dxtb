"""
Integral configuration.
"""
from __future__ import annotations

from .._types import Literal
from ..constants import defaults, labels

__all__ = ["ConfigIntegrals"]


class ConfigIntegrals:
    """
    Configuration for the integrals.
    """

    cutoff: float
    """Real-space cutoff (in Bohr) for integral evaluation for PyTorch."""

    driver: int
    """Type of integral driver."""

    uplo: Literal["n", "l", "u"]
    """Integral mode for PyTorch integral calculation."""

    def __init__(
        self,
        *,
        cutoff=defaults.INTCUTOFF,
        driver=defaults.INTDRIVER,
        uplo=defaults.INTUPLO,
    ) -> None:
        self.cutoff = cutoff

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

        if driver.casefold() in ("libcint", "c"):
            self.driver = labels.INTDRIVER_LIBCINT
        elif driver.casefold() in ("pytorch", "torch", "dxtb"):
            self.driver = labels.INTDRIVER_PYTORCH
        elif driver.casefold() in ("pytorch2", "torch2", "dxtb2"):
            self.driver = labels.INTDRIVER_PYTORCH2
        else:
            raise ValueError(f"Unknown guess method '{driver}'.")
