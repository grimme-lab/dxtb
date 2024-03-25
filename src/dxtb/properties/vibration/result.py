"""
Vibrational Analysis: Result Container
======================================

Base class for vibrational analysis results.
"""

from __future__ import annotations

import torch
from tad_mctc.typing import NoReturn, Tensor, TensorLike
from tad_mctc.units import AU2RCM


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
            "Setting IR frequencies is not supported. Iternally, the "
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

    def __getitem__(self, key: str) -> Tensor:
        if key not in self.__slots__:
            key = f"_{key}"

        if key in self.__slots__:
            return getattr(self, key)

        raise KeyError(f"Invalid key: {key}")

    def __str__(self) -> str:
        text = ""
        for s in self.__slots__:
            if "unit" in s:
                text += s.replace("_unit", "").replace("_", "")
                text += f": {getattr(self, s)}, "

        # remove last comma and space
        text = text[:-2]
        return f"{self.__class__.__name__}({text})"

    def __repr__(self) -> str:
        return str(self)
