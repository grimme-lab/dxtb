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
Integrals: Wrappers/Shortcuts
=============================

A simple collection of convenience functions to obtain all integral matrices.
This is intended for testing and developing. In these functions, defaults will
be applied. Although (some) settings can be accessed through keyword arguments,
it is recommended to follow the interal integral builds as used in the
:class:`~dxtb.integrals.Integrals` class for more direct control.

Note that there are several peculiarities for the multipole integrals:

- The multipole operators are centered on ``(0, 0, 0)`` (r0) and not on the ket
  (rj), the latter being the default in ``dxtb``.
- An overlap calculation is executed for the normalization of the multipole
  integral every time :func:`.dipole` or :func:`.quadrupole` are called.
- The quadrupole integral is **not** in its traceless representation.

Example
-------

.. code-block:: python

    from dxtb.integrals.wrappers import overlap, dipint, quadint
    from dxtb import GFN1_XTB as par
    import torch

    numbers = torch.tensor([14, 1, 1, 1, 1])
    positions = torch.tensor([
        [0.00000000000000, 0.00000000000000, 0.00000000000000],
        [1.61768389755830, 1.61768389755830, -1.61768389755830],
        [-1.61768389755830, -1.61768389755830, -1.61768389755830],
        [1.61768389755830, -1.61768389755830, 1.61768389755830],
        [-1.61768389755830, 1.61768389755830, 1.61768389755830],
    ])

    # Calculate the overlap integrals using the GFN1_XTB parameters
    s = overlap(numbers, positions, par)
    print(s.shape)  # Output: torch.Size([17, 17])

    # Calculate the dipole integrals
    d = dipint(numbers, positions, par)
    print(d.shape)  # Output: torch.Size([3, 17, 17])

    # Calculate the quadrupole integrals
    q = quadint(numbers, positions, par)
    print(q.shape)  # Output: torch.Size([9, 17, 17])
"""

from __future__ import annotations

from dxtb import IndexHelper
from dxtb._src.constants import labels
from dxtb._src.param import Param
from dxtb._src.typing import DD, Any, Literal, Tensor
from dxtb._src.xtb.gfn1 import GFN1Hamiltonian
from dxtb._src.xtb.gfn2 import GFN2Hamiltonian

from .factory import new_driver
from .types import Dipole, Overlap, Quadrupole

__all__ = ["hcore", "overlap", "dipint", "quadint"]


def hcore(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for the core Hamiltonian matrix calculation.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Core Hamiltonian matrix of shape ``(nb, nao, nao)``.

    Raises
    ------
    TypeError
        If the Hamiltonian parametrization does not contain meta data or if the
        meta data does not contain a name to select the correct Hamiltonian.
    ValueError
        If the Hamiltonian name is unknown.
    """
    if par.meta is None:
        raise TypeError(
            "Meta data of Hamiltonian parametrization must contain a name. "
            "Otherwise, the correct Hamiltonian cannot be selected internally."
        )

    if par.meta.name is None:
        raise TypeError(
            "The name field of the meta data of the Hamiltonian "
            "parametrization must contain a name. Otherwise, the correct "
            "Hamiltonian cannot be selected internally."
        )

    dd: DD = {"device": positions.device, "dtype": positions.dtype}
    ihelp = IndexHelper.from_numbers(numbers, par)

    name = par.meta.name.casefold()
    if name == "gfn1-xtb":
        h0 = GFN1Hamiltonian(numbers, par, ihelp, **dd)
    elif name == "gfn2-xtb":
        h0 = GFN2Hamiltonian(numbers, par, ihelp, **dd)
    else:
        raise ValueError(f"Unknown Hamiltonian type '{name}'.")

    # TODOGFN2: Handle possibly different CNs
    cn = kwargs.pop("cn", None)
    if cn is None:
        # pylint: disable=import-outside-toplevel
        from ..ncoord import cn_d3

        cn = cn_d3(numbers, positions)

    ovlp = overlap(numbers, positions, par)
    return h0.build(positions, ovlp, cn=cn)


def overlap(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for overlap integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Overlap integral (matrix) of shape ``(nb, nao, nao)``.
    """
    return _integral("_overlap", numbers, positions, par, **kwargs)


def dipint(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for dipole integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Dipole integral (matrix) of shape ``(nb, 3, nao, nao)``.
    """
    return _integral("_dipole", numbers, positions, par, **kwargs)


def quadint(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """Shortcut for dipole integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Dipole integral (matrix) of shape ``(nb, 3, nao, nao)``.
    """
    return _integral("_quadrupole", numbers, positions, par, **kwargs)


def _integral(
    integral_type: Literal["_overlap", "_dipole", "_quadrupole"],
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    **kwargs: Any,
) -> Tensor:
    """Shortcut for integral calculations.

    Parameters
    ----------
    integral_type : Literal['overlap', 'dipole', 'quadrupole']
        Type of integral to calculate.
    numbers : Tensor
        Atomic numbers for all atoms in the system (shape: ``(..., nat)``).
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Integral matrix of shape ``(nb, nao, nao)`` for overlap and
        ``(nb, 3, nao, nao)`` for dipole and quadrupole.

    Raises
    ------
    ValueError
        If the integral type is unknown.
    """
    dd: DD = {"device": positions.device, "dtype": positions.dtype}

    # Determine which driver class to instantiate (defaults to libcint)
    driver_name = kwargs.pop("driver", labels.INTDRIVER_LIBCINT)
    driver = new_driver(driver_name, numbers, par, **dd)

    # setup driver for integral calculation
    driver.setup(positions)

    # inject driver into requested integral
    if integral_type == "_overlap":
        integral = Overlap(driver=driver_name, **dd, **kwargs)
    elif integral_type in ("_dipole", "_quadrupole"):
        ovlp = Overlap(driver=driver_name, **dd, **kwargs)

        # multipole integrals require the overlap for normalization
        if ovlp.integral._matrix is None or ovlp.integral._norm is None:
            ovlp.build(driver)

        if integral_type == "_dipole":
            integral = Dipole(driver=driver_name, **dd, **kwargs)
        elif integral_type == "_quadrupole":
            integral = Quadrupole(driver=driver_name, **dd, **kwargs)

        integral.integral.norm = ovlp.integral.norm
    else:
        raise ValueError(f"Unknown integral type '{integral_type}'.")

    # actual integral calculation
    return integral.build(driver)
