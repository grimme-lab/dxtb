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

    level: int
    """
    Indicator for integrals to compute.
    - 0: None
    - 1: overlap
    - 2: +dipole
    - 3: +quadrupole
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
        level: int = defaults.INTLEVEL,
        cutoff: float = defaults.INTCUTOFF,
        driver: str | int = defaults.INTDRIVER,
        uplo: str = defaults.INTUPLO,
    ) -> None:
        self.cutoff = cutoff

        if not isinstance(level, int):
            raise TypeError(
                f"The received integral level (`{level}`) is not an integer, "
                f"but {type(level)}."
            )
        self.level = level

        if uplo not in ("n", "N", "u", "U", "l", "L"):
            raise ValueError(f"Unknown option for `uplo` chosen: '{uplo}'.")
        self.uplo = uplo.casefold()  # type: ignore

        if isinstance(driver, str):
            if driver.casefold() in labels.INTDRIVER_LIBCINT_STRS:
                self.driver = labels.INTDRIVER_LIBCINT
            elif driver.casefold() in labels.INTDRIVER_PYTORCH_STRS:
                self.driver = labels.INTDRIVER_PYTORCH
            elif driver.casefold() in labels.INTDRIVER_PYTORCH2_STRS:
                self.driver = labels.INTDRIVER_PYTORCH2
            else:
                raise ValueError(f"Unknown integral driver '{driver}'.")
        elif isinstance(driver, int):
            if driver not in (
                labels.INTDRIVER_LIBCINT,
                labels.INTDRIVER_PYTORCH,
                labels.INTDRIVER_PYTORCH2,
            ):
                raise ValueError(f"Unknown integral driver '{driver}'.")

            self.driver = driver
        else:
            raise TypeError(
                "The driver must be of type 'int' or 'str', but "
                f"'{type(driver)}' was given."
            )
