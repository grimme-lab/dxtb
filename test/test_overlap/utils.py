"""
Utility function for overlap calculation.
"""

from __future__ import annotations

from dxtb._types import DD, Literal, Tensor
from dxtb.basis import IndexHelper
from dxtb.integral.driver.pytorch import IntDriverPytorch as IntDriver
from dxtb.integral.driver.pytorch import OverlapPytorch as Overlap
from dxtb.param import Param


def calc_overlap(
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    dd: DD,
    uplo: Literal["n", "N", "u", "U", "l", "L"] = "l",
) -> Tensor:
    ihelp = IndexHelper.from_numbers(numbers, par)
    driver = IntDriver(numbers, par, ihelp, **dd)
    overlap = Overlap(uplo=uplo, **dd)

    driver.setup(positions)
    return overlap.build(driver)
