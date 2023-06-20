"""
Type annotations
================

This module contains all type annotations for this project.

Since typing still significantly changes across different Python versions,
all the special cases are handled here as well.
"""
# pylint: disable=unused-import
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, TypedDict, overload

import torch
from torch import Tensor

from .constants import defaults

# "Self" (since Python 3.11)
if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

# "TypeGuard" (since Python 3.10)
if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

# starting with Python 3.9, type hinting generics have been moved
# from the "typing" to the "collections" module
# (see PEP 585: https://peps.python.org/pep-0585/)
if sys.version_info >= (3, 9):
    from collections.abc import Callable, Generator, Sequence
else:
    from typing import Callable, Generator, Sequence

# type aliases that do not require "from __future__ import annotations"
CountingFunction = Callable[[Tensor, Tensor], Tensor]
Gather = Callable[[Tensor, int, Tensor], Tensor]
Scatter = Callable[[Tensor, int, Tensor, str], Tensor]

if sys.version_info >= (3, 10):
    # "from __future__ import annotations" only affects type annotations
    # not type aliases, hence "|" is not allowed before Python 3.10
    PathLike = str | Path
    ScatterOrGather = Gather | Scatter
    Slicer = list[slice] | tuple[slice] | tuple[Ellipsis]
    Size = list[Tensor] | list[int] | tuple[int] | torch.Size
    TensorOrTensors = list[Tensor] | tuple[Tensor, ...] | Tensor
elif sys.version_info >= (3, 9):
    # in Python 3.9, "from __future__ import annotations" works with type
    # aliases but requires using `Union` from typing
    from typing import Union

    PathLike = Union[str, Path]
    ScatterOrGather = Union[Gather, Scatter]
    Slicer = Union[list[slice], tuple[slice], tuple[Ellipsis]]
    Size = Union[list[Tensor], list[int], tuple[int], torch.Size]
    TensorOrTensors = Union[list[Tensor], tuple[Tensor, ...], Tensor]
elif sys.version_info >= (3, 8):
    # in Python 3.8, "from __future__ import annotations" only affects
    # type annotations not type aliases
    from typing import List, Tuple, Union

    PathLike = Union[str, Path]
    ScatterOrGather = Union[Gather, Scatter]
    Slicer = Union[List[slice], Tuple[slice], Tuple]
    Size = Union[List[Tensor], List[int], Tuple[int], torch.Size]
    TensorOrTensors = Union[List[Tensor], Tuple[Tensor, ...], Tensor]
else:
    raise RuntimeError(
        f"'dxtb' requires at least Python 3.8 (Python {sys.version_info.major}."
        f"{sys.version_info.minor}.{sys.version_info.micro} found)."
    )


class Slicers(TypedDict):
    """Collection of slicers of different resolutions for culling in SCF."""

    orbital: Slicer
    """Slicer for orbital-resolved variables."""
    shell: Slicer
    """Slicer for shell-resolved variables."""
    atom: Slicer
    """Slicer for atom-resolved variables."""


class SCFResult(TypedDict):
    """Collection of SCF result variables."""

    charges: Tensor
    """Self-consistent orbital-resolved Mulliken partial charges."""

    coefficients: Tensor
    """LCAO-MO coefficients (eigenvectors of Fockian)."""

    density: Tensor
    """Density matrix."""

    emo: Tensor
    """Energy of molecular orbitals (sorted by increasing energy)."""

    energy: Tensor
    """Energies of the self-consistent contributions (interactions)."""

    fenergy: Tensor
    """Atom-resolved electronic free energy from fractional occupation."""

    hamiltonian: Tensor
    """Full Hamiltonian matrix (H0 + H1)."""

    occupation: Tensor
    """Orbital occupations."""

    potential: Tensor
    """Self-consistent orbital-resolved potential."""


class Molecule(TypedDict):
    """
    Representation of fundamental molecular structure (atom types and postions).
    """

    numbers: Tensor
    """Tensor of atomic numbers"""

    positions: Tensor
    """Tensor of 3D coordinates of shape (n, 3)"""


class TensorLike:
    """
    Provide `device` and `dtype` as well as `to()` and `type()` for other
    classes.

    The selection of `torch.Tensor` variables to change within the class is
    handled by searching `__slots__`. Hence, if one wants to use this
    functionality the subclass of `TensorLike` must specify `__slots__`.
    """

    __slots__ = ["__device", "__dtype"]

    def __init__(
        self, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        self.__device = (
            device if device is not None else torch.device(defaults.TORCH_DEVICE)
        )
        self.__dtype = dtype if dtype is not None else defaults.TORCH_DTYPE

    @property
    def device(self) -> torch.device:
        """The device on which the class object resides."""
        return self.__device

    @device.setter
    def device(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the ".to" method if wanting to change device.

        Returns
        -------
        NoReturn
            Always raises an `AttributeError`.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by class object."""
        return self.__dtype

    @dtype.setter
    def dtype(self, *_: Any) -> NoReturn:
        """
        Instruct users to use the `.type` method if wanting to change dtype.

        Returns
        -------
        NoReturn
            Always raises an `AttributeError`.

        Raises
        ------
        AttributeError
            Setter is called.
        """
        raise AttributeError("Change object to dtype using the `.type` method")

    def type(self, dtype: torch.dtype) -> Self | NoReturn:
        """
        Returns a copy of the `TensorLike` instance with specified floating
        point type.
        This method creates and returns a new copy of the `TensorLike` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Floating point type.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance with the specified dtype.

        Notes
        -----
        If the `TensorLike` instance has already the desired dtype `self` will
        be returned.
        """
        if self.dtype == dtype:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `type` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        if dtype not in self.allowed_dtypes:
            raise ValueError(
                f"Only '{self.allowed_dtypes}' allowed (received '{dtype}')."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    if attr.dtype in self.allowed_dtypes:
                        attr = attr.type(dtype)  # type: ignore
                args[s] = attr

        return self.__class__(**args, dtype=dtype)

    def to(self, device: torch.device) -> Self | NoReturn:
        """
        Returns a copy of the `TensorLike` instance on the specified device.

        This method creates and returns a new copy of the `TensorLike` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        TensorLike
            A copy of the `TensorLike` instance placed on the specified device.

        Notes
        -----
        If the `TensorLike` instance is already on the desired device `self`
        will be returned.
        """
        if self.device == device:
            return self

        if len(self.__slots__) == 0:
            raise RuntimeError(
                f"The `to` method requires setting `__slots__` in the "
                f"'{self.__class__.__name__}' class."
            )

        args = {}
        for s in self.__slots__:
            if not s.startswith("__"):
                attr = getattr(self, s)
                if isinstance(attr, Tensor) or issubclass(type(attr), TensorLike):
                    attr = attr.to(device=device)  # type: ignore
                args[s] = attr

        return self.__class__(**args, device=device)

    @property
    def allowed_dtypes(self) -> tuple[torch.dtype, ...]:
        """
        Specification of dtypes that the TensorLike object can take. Defaults
        to float types and must be overridden by subclass if float are not
        allowed. The IndexHelper is an example that should only allow integers.

        Returns
        -------
        tuple[torch.dtype, ...]
            Collection of allowed dtypes the TensorLike object can take.
        """
        return (torch.float16, torch.float32, torch.float64)
