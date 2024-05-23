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
Vibrational Analysis: IR Spectra
================================

Calculate IR intensities from the geometric dipole derivative.
"""

from __future__ import annotations

import torch
from tad_mctc.math import einsum
from tad_mctc.units import AU2KMMOL

from dxtb._src.typing import Any, Literal, NoReturn, Tensor

from .result import BaseResult

__all__ = ["ir_ints", "IRResult"]


class IRResult(BaseResult):
    """
    Data from the calculation of an IR spectrum.

    - Vibrational frequencies
    - IR intensities
    """

    converter_ints: dict[str, float] = {
        "a.u.": 1.0,
        "km/mol": AU2KMMOL,
    }

    __slots__ = ["_ints", "_ints_unit"]

    def __init__(
        self,
        freqs: Tensor,
        ints: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the IR result.

        Parameters
        ----------
        freqs : Tensor
            Vibrational frequencies in atomic units.
        ints : Tensor
            IR intensities in atomic units.
        device : torch.device | None, optional
            Device of the tensors. If ``None``, the device of `freqs` is used.
            Defaults to ``None``.
        dtype : torch.dtype | None, optional
            Data type of the tensors. If ``None``, the data type of `freqs` is
            used. Defaults to ``None``.
        """
        super().__init__(
            freqs=freqs,
            device=device if device is not None else freqs.device,
            dtype=dtype if dtype is not None else freqs.dtype,
        )

        self._ints = ints
        self._ints_unit = "a.u."

    # intensities

    @property
    def ints(self) -> Tensor:
        return self._ints * self.converter_ints[self._ints_unit]

    @ints.setter
    def ints(self, *_: Any) -> NoReturn:
        raise RuntimeError(
            "Setting IR intensities is not supported. Internally, the "
            "intensities should always be stored in atomic units. Use "
            "the `to_unit` method to convert to a different unit or set the "
            "`ints_unit` attribute."
        )

    @property
    def ints_unit(self) -> str:
        return self._ints_unit

    @ints_unit.setter
    def ints_unit(self, value: str) -> None:
        if value not in self.converter_ints:
            raise ValueError(f"Unsupported intensity unit: {value}")
        self._ints_unit = value

    # conversion

    def to_unit(self, value: Literal["freqs", "ints"], unit: str) -> Tensor:
        """
        Convert a value from one unit to another based on the converter dictionary.
        """
        if value == "freqs":
            return self._convert(self.freqs, unit, self.converter_freqs)

        if value == "ints":
            return self._convert(self.ints, unit, self.converter_ints)

        raise ValueError(f"Unsupported value for conversion: {value}")

    def use_common_units(self) -> None:
        """
        Convert the frequencies and intensities to common units, that is,
        `cm^-1` for frequencies and `km/mol` for intensities.
        """
        self.freqs_unit = "cm^-1"
        self.ints_unit = "km/mol"


def ir_ints(dmu_dr: Tensor, modes: Tensor) -> Tensor:
    """
    Calculate IR intensities from the geometric dipole derivative.

    Parameters
    ----------
    dmu_dr : Tensor
        Dipole derivative tensor of shape `(..., 3, nat, 3)`.
    modes : Tensor
        Normal modes of shape `(..., nat*3, nmodes)`.

    Returns
    -------
    Tensor
        IR intensities of shape `(..., nfreqs)`.
    """
    # reshape for matmul: (..., 3, nat, 3) -> (..., 3, nat*3)
    dmu_dr = dmu_dr.view(*(*modes.shape[:-2], 3, -1))

    # convert cartesian to internal coordinate derivatives
    # (..., 3, nat*3) @ (..., nat*3, nfreqs) = (..., 3, nfreqs)
    dmu_dq = dmu_dr @ modes

    # square deriv and sum along cartesian components for intensity
    # (..., 3, nfreqs) * (..., 3, nfreqs) -> (..., nfreqs)
    return einsum("...xf,...xf->...f", dmu_dq, dmu_dq)
