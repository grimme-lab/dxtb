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
Factories
=========

Factory functions for integral classes.
"""

from __future__ import annotations

import torch

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param
from dxtb._src.typing import TYPE_CHECKING, Any, Tensor

if TYPE_CHECKING:
    from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
    from dxtb._src.xtb.gfn2 import GFN2Hamiltonian

    from .driver.libcint import DipoleLibcint, OverlapLibcint, QuadrupoleLibcint
    from .driver.pytorch import DipolePytorch, OverlapPytorch, QuadrupolePytorch
    from .types import DipoleIntegral, OverlapIntegral, QuadrupoleIntegral

__all__ = ["new_hcore", "new_overlap", "new_dipint", "new_quadint"]


################################################################################


def new_hcore(
    numbers: Tensor,
    par: Param,
    ihelp: IndexHelper,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GFN1Hamiltonian | GFN2Hamiltonian:
    if par.meta is None:
        raise ValueError(
            "The `meta` information field is missing in the parametrization. "
            "No xTB core Hamiltonian can be selected and instantiated."
        )

    if par.meta.name is None:
        raise ValueError(
            "The `name` field of the meta information is missing in the "
            "parametrization. No xTB core Hamiltonian can be selected and "
            "instantiated."
        )

    if par.meta.name.casefold() in ("gfn1-xtb", "gfn1"):
        return new_hcore_gfn1(numbers, ihelp, par, device=device, dtype=dtype)

    if par.meta.name.casefold() in ("gfn2-xtb", "gfn2"):
        return new_hcore_gfn2(numbers, ihelp, par, device=device, dtype=dtype)

    raise ValueError(f"Unsupported Hamiltonian type: {par.meta.name}")


def new_hcore_gfn1(
    numbers: Tensor,
    ihelp: IndexHelper,
    par: Param | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GFN1Hamiltonian:
    # pylint: disable=import-outside-toplevel
    from dxtb._src.xtb.gfn1 import GFN1Hamiltonian as Hamiltonian

    if par is None:
        # pylint: disable=import-outside-toplevel
        from dxtb import GFN1_XTB as par

    return Hamiltonian(numbers, par, ihelp, device=device, dtype=dtype)


def new_hcore_gfn2(
    numbers: Tensor,
    ihelp: IndexHelper,
    par: Param | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> GFN2Hamiltonian:
    # pylint: disable=import-outside-toplevel
    from dxtb._src.xtb.gfn2 import GFN2Hamiltonian as Hamiltonian

    if par is None:
        # pylint: disable=import-outside-toplevel
        from dxtb import GFN2_XTB as par

    return Hamiltonian(numbers, par, ihelp, device=device, dtype=dtype)


################################################################################


def new_overlap(
    driver: int = labels.INTDRIVER_LIBCINT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapIntegral:
    # Determine which integral class to instantiate based on the type
    if driver == labels.INTDRIVER_LIBCINT:
        return new_overlap_libcint(device=device, dtype=dtype, **kwargs)

    if driver in (
        labels.INTDRIVER_ANALYTICAL,
        labels.INTDRIVER_AUTOGRAD,
        labels.INTDRIVER_LEGACY,
    ):
        return new_overlap_pytorch(device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown integral driver '{driver}'.")


def new_overlap_libcint(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapLibcint:
    # pylint: disable=import-outside-toplevel
    from .driver.libcint import OverlapLibcint as Overlap

    if kwargs.pop("force_cpu_for_libcint", True):
        device = torch.device("cpu")

    return Overlap(device=device, dtype=dtype, **kwargs)


def new_overlap_pytorch(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> OverlapPytorch:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import OverlapPytorch as Overlap

    return Overlap(device=device, dtype=dtype, **kwargs)


################################################################################


def new_dipint(
    driver: int = labels.INTDRIVER_LIBCINT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> DipoleIntegral:
    # Determine which integral class to instantiate based on the type
    if driver == labels.INTDRIVER_LIBCINT:
        return new_dipint_libcint(device=device, dtype=dtype, **kwargs)

    if driver in (
        labels.INTDRIVER_ANALYTICAL,
        labels.INTDRIVER_AUTOGRAD,
        labels.INTDRIVER_LEGACY,
    ):
        return new_dipint_pytorch(device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown integral driver '{driver}'.")


def new_dipint_libcint(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> DipoleLibcint:
    # pylint: disable=import-outside-toplevel
    from .driver.libcint import DipoleLibcint as _Dipole

    if kwargs.pop("force_cpu_for_libcint", True):
        device = torch.device("cpu")

    return _Dipole(device=device, dtype=dtype, **kwargs)


def new_dipint_pytorch(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> DipolePytorch:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import DipolePytorch as _Dipole

    return _Dipole(device=device, dtype=dtype, **kwargs)


################################################################################


def new_quadint(
    driver: int = labels.INTDRIVER_LIBCINT,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> QuadrupoleIntegral:
    # Determine which integral class to instantiate based on the type
    if driver == labels.INTDRIVER_LIBCINT:
        return new_quadint_libcint(device=device, dtype=dtype, **kwargs)

    if driver in (
        labels.INTDRIVER_ANALYTICAL,
        labels.INTDRIVER_AUTOGRAD,
        labels.INTDRIVER_LEGACY,
    ):
        return new_quadint_pytorch(device=device, dtype=dtype, **kwargs)

    raise ValueError(f"Unknown integral driver '{driver}'.")


def new_quadint_libcint(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> QuadrupoleLibcint:
    # pylint: disable=import-outside-toplevel
    from .driver.libcint import QuadrupoleLibcint as Quadrupole

    if kwargs.pop("force_cpu_for_libcint", True):
        device = torch.device("cpu")

    return Quadrupole(device=device, dtype=dtype, **kwargs)


def new_quadint_pytorch(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    **kwargs: Any,
) -> QuadrupolePytorch:
    # pylint: disable=import-outside-toplevel
    from .driver.pytorch import QuadrupolePytorch as Quadrupole

    return Quadrupole(device=device, dtype=dtype, **kwargs)
