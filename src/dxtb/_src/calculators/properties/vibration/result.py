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
Vibrational Analysis: Result Container
======================================

Base class for vibrational analysis results.
"""

from __future__ import annotations

import torch
from tad_mctc.units import AU2RCM

from dxtb._src.typing import Generator, NoReturn, PathLike, Tensor, TensorLike
from dxtb._src.utils.misc import get_all_slots

__all__ = ["BaseResult"]


class BaseResult(TensorLike):
    """
    Base class for vibrational analysis results.
    Vibrational results always stored the frequencies (in atomic units).
    """

    converter_freqs: dict[str, float] = {
        "a.u.": 1.0,
        "cm^-1": AU2RCM,
    }

    __slots__ = ["_freqs", "_freqs_unit"]

    def __init__(
        self,
        freqs: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=device if device is not None else freqs.device,
            dtype=dtype if dtype is not None else freqs.dtype,
        )

        self._freqs = freqs
        self._freqs_unit = "a.u."

    # frequencies

    @property
    def freqs(self) -> Tensor:
        return self._freqs * self.converter_freqs[self.freqs_unit]

    @freqs.setter
    def freqs(self, *_) -> NoReturn:
        raise RuntimeError(
            "Setting IR frequencies is not supported. Internally, the "
            "frequencies should always be stored in atomic units. Use "
            "the `to_unit` method to convert to a different unit or set the "
            "`freqs_unit` attribute."
        )

    @property
    def freqs_unit(self) -> str:
        return self._freqs_unit

    @freqs_unit.setter
    def freqs_unit(self, value: str) -> None:
        if value not in self.converter_freqs:
            raise ValueError(f"Unsupported frequency unit: {value}")
        self._freqs_unit = value

    def _convert(self, value: Tensor, unit: str, converter: dict[str, float]) -> Tensor:
        """
        Convert a tensor from one unit to another based on the converter
        dictionary.

        Parameters
        ----------
        value : Tensor
            The tensor to convert.
        unit : str
            The unit to convert to.
        converter : dict[str, float]
            The dictionary with the conversion factors.

        Returns
        -------
        Tensor
            The converted tensor.

        Raises
        ------
        ValueError
            If the unit is not supported.
        """
        if unit not in converter:
            raise ValueError(f"Unsupported unit for conversion: {unit}")

        return value * converter[unit]

    def save_prop_to_pt(self, prop: str, filepath: PathLike | None = None) -> None:
        """
        Save the results to a PyTorch file.

        Parameters
        ----------
        prop : str
            The property to save.
        filepath : PathLike
            Path to save the results to.
        """
        s = get_all_slots(self)
        if prop not in s:
            # remove underscore
            s = [i[1:] for i in s]
            if prop not in s:
                raise ValueError(f"Invalid property: {prop}")

        # use custom __getitem__ method
        tensor = self[prop]

        if filepath is None:
            name = self.__class__.__name__.casefold().replace("result", "")
            filepath = f"{name}-{prop.replace('_', '')}.pt"

        torch.save(tensor.detach(), filepath)

    def save_all_to_pt(self, filepaths: list[PathLike] | None = None) -> None:
        """
        Save all results to a PyTorch file (".pt").

        Parameters
        ----------
        filepath : PathLike
            Path to save the results to.
        """
        s = get_all_slots(self)
        paths = [None] * len(s) if filepaths is None else filepaths

        for prop, path in zip(s, paths):
            if "unit" not in prop:
                self.save_prop_to_pt(prop, path)

    def __iter__(self) -> Generator[Tensor, None, None]:
        return iter(getattr(self, s) for s in get_all_slots(self) if "unit" not in s)

    def __getitem__(self, key: str) -> Tensor:
        s = get_all_slots(self)

        # Check if key is a property first (properties won't be in __slots__)
        if hasattr(self.__class__, key):
            if isinstance(getattr(self.__class__, key), property):
                return getattr(self, key)

        if key not in s:
            key = f"_{key}"

        if key in s:
            return getattr(self, key)

        raise KeyError(f"Invalid key: '{key}'. Possible keys are '{', '.join(s)}'.")

    def __str__(self) -> str:
        text = ""
        for s in get_all_slots(self):
            attr = getattr(self, s)

            if "unit" not in s:
                attr = attr.shape

            if s.startswith("_"):
                s = s[1:]

            text += f"{s}: {attr}, "

        # remove last comma and space
        text = text[:-2]
        return f"{self.__class__.__name__}({text})"

    def __repr__(self) -> str:
        return str(self)
