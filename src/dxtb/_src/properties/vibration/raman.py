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
Vibrational Analysis: Raman Spectra
===================================

Calculate Raman intensities and the depolarization ratio from the geometric
polarizability derivative.
"""

from __future__ import annotations

import torch
from tad_mctc import storch
from tad_mctc.math import einsum
from tad_mctc.units import AU2AA4AMU

from dxtb._src.typing import Any, Literal, NoReturn, Tensor

from .result import BaseResult

__all__ = ["raman_ints_depol", "RamanResult"]


class RamanResult(BaseResult):
    """
    Data from the calculation of a Raman spectrum.

    - Vibrational frequencies
    - Raman activities (intensities)
    - Depolarization ratios
    """

    converter_ints: dict[str, float] = {
        "a.u.": 1.0,
        "A^4/amu": AU2AA4AMU,
    }

    __slots__ = ["_ints", "_ints_unit", "_depol", "_depol_unit"]

    def __init__(
        self,
        freqs: Tensor,
        ints: Tensor,
        depol: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize the Raman result.

        Parameters
        ----------
        freqs : Tensor
            Vibrational frequencies in atomic units.
        ints : Tensor
            IR intensities (activities) in atomic units.
        depol : Tensor
            Depolarization ratio (unitless).
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

        self._depol = depol
        self._depol_unit = None

    # intensities

    @property
    def ints(self) -> Tensor:
        return self._ints * self.converter_ints[self._ints_unit]

    @ints.setter
    def ints(self, *_: Any) -> NoReturn:
        raise RuntimeError(
            "Setting Raman intensities (activities) is not supported. "
            "Internally, the intensities should always be stored in atomic "
            "units. Use the `to_unit` method to convert to a different unit "
            "or set the `ints_unit` attribute."
        )

    @property
    def ints_unit(self) -> str:
        return self._ints_unit

    @ints_unit.setter
    def ints_unit(self, value: str) -> None:
        if value not in self.converter_ints:
            raise ValueError(f"Unsupported intensity unit: {value}")
        self._ints_unit = value

    # depolarization ratio

    @property
    def depol(self) -> Tensor:
        return self._depol

    @depol.setter
    def depol(self, *_: Any) -> NoReturn:
        raise RuntimeError("Setting depolarization ratios is not supported.")

    @property
    def depol_unit(self) -> None:
        return self._depol_unit

    @depol_unit.setter
    def depol_unit(self, *_: Any) -> NoReturn:
        raise RuntimeError("The depolarization ratios are unitless.")

    # conversion

    def to_unit(self, value: Literal["freqs", "ints", "depol"], unit: str) -> Tensor:
        """
        Convert a value from one unit to another based on the converter
        dictionary.

        Parameters
        ----------
        value : Literal['freqs', 'ints', 'depol']
            The value (stored property) to convert.
        unit : str
            The unit to convert to.

        Returns
        -------
        Tensor
            The converted value.

        Raises
        ------
        NotImplementedError
            If the value is "depol", because the depolarization ratio is
            unitless.
        ValueError
            If the value (name of the stored property) does not exist.
        """
        if value == "freqs":
            return self._convert(self.freqs, unit, self.converter_freqs)

        if value == "ints":
            return self._convert(self.ints, unit, self.converter_ints)

        if value == "depol":
            raise NotImplementedError("Depolarization ratio is unitless.")

        raise ValueError(f"Unsupported value for conversion: {value}")

    def use_common_units(self) -> None:
        """
        Convert the frequencies and intensities to common units, that is,
        `cm^-1` for frequencies and `A^4/amu` for intensities.
        """
        self.freqs_unit = "cm^-1"
        self.ints_unit = "A^4/amu"


def raman_ints_depol(da_dr: Tensor, modes: Tensor) -> tuple[Tensor, Tensor]:
    r"""
    Calculate static Raman activities (intensities) and the depolarization
    ratios from the geometric polarizability derivative (Raman susceptibility
    tensor :math:`\chi`).

    Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

    Parameters
    ----------
    da_dr : Tensor
        Geometric polarizability derivative tensor of shape `(..., 3, 3, nat, 3)`.
    modes : Tensor
        Normal modes of shape `(..., nat*3, nmodes)`.

    Returns
    -------
    tuple[Tensor, Tensor]
        Static Raman activities (intensities) of shape `(..., nfreqs)` and
        depolarization ratios of shape `(..., nfreqs)`.
    """
    # reshape for matmul: (..., 3, 3, nat, 3) -> (..., 3, 3, nat*3)
    da_dr = da_dr.view(*(*modes.shape[:-2], 3, 3, -1))

    # convert cartesian to internal coordinate derivatives
    # (..., 3, 3, nat*3) @ (..., nat*3, nmodes) = (..., 3, 3, nmodes)
    da_dq = da_dr @ modes

    # Eq.3 with alpha' = a (trace of the polarizability derivative)
    a = einsum("...iij->...j", da_dq)
    a2 = torch.pow(a, 2.0)

    # Eq.4 with (gamma')^2 = g = 0.5 * (g1 + g2 + g3 + 6.0*g4)
    g1 = (da_dq[0, 0] - da_dq[1, 1]) ** 2
    g2 = (da_dq[0, 0] - da_dq[2, 2]) ** 2
    g3 = (da_dq[2, 2] - da_dq[1, 1]) ** 2
    g4 = da_dq[0, 1] ** 2 + da_dq[1, 2] ** 2 + da_dq[2, 0] ** 2
    g = g1 + g2 + g3 + 6.0 * g4

    # Eq.1 (the 1/3 from Eq.3 is squared, yielding 45 * 1/9 = 5; the 7 is
    # halfed by the 0.5 from Eq.4)
    ints = 5 * a2 + 3.5 * g

    # original formula: 3 * gamma^2 / (45 * alpha^2 + 4 * gamma^2)
    depol = torch.where(
        a2 > 1e-8,  # avoid division by tiny values...verify correctness?
        storch.divide(1.5 * g, 5 * a2 + 2.0 * g),
        torch.tensor(0.0, device=a.device, dtype=a.dtype),
    )

    return ints, depol
