"""
Wrappers/Shortcuts for Integrals
================================

A simple collection for convenience functions of all integrals. In these
functions, defaults will be applied. Although (some) settings can be accessed
through keyword arguments, it is recommended to follow the interal integral
builds as used in the :class:`.Integrals` class for more direct control.

Note that there are several peculiarities for the multipole integrals:
- The multipole operators are centered on `(0, 0, 0)` (r0) and not on ket (rj),
  the latter being the default in `dxtb`.
- An overlap calculation is executed for the normalization of the multipole
  integral every time :func:`.dipole` or :func:`.quadrupole` are called.
- The quadrupole integral is not in its traceless representation.

Example
-------
>>> from dxtb.integral.wrappers import overlap, dipole, quadrupole
>>> from dxtb.param import GFN1_XTB as par
>>>
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
     [+0.00000000000000, +0.00000000000000, +0.00000000000000],
     [+1.61768389755830, +1.61768389755830, -1.61768389755830],
     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
     [+1.61768389755830, -1.61768389755830, +1.61768389755830],
     [-1.61768389755830, +1.61768389755830, +1.61768389755830],
 ])
>>>
>>> s = overlap(numbers, positions, par)
>>> print(s.shape)
torch.Size([17, 17])
>>> d = dipole(numbers, positions, par)
>>> print(d.shape)
torch.Size([3, 17, 17])
>>> q = quadrupole(numbers, positions, par)
>>> print(q.shape)
torch.Size([9, 17, 17])
"""

from __future__ import annotations

from .._types import DD, Any, Literal, Tensor
from ..basis import IndexHelper
from ..constants import labels
from ..param import Param, get_elem_angular
from ..xtb.h0_gfn1 import GFN1Hamiltonian
from ..xtb.h0_gfn2 import GFN2Hamiltonian
from .dipole import Dipole
from .factory import new_driver
from .overlap import Overlap
from .quadrupole import Quadrupole

__all__ = ["hcore", "overlap", "dipole", "quadrupole"]


def hcore(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
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
    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))

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
        from ..ncoord import exp_count, get_coordination_number

        cn = get_coordination_number(numbers, positions, exp_count)

    ovlp = overlap(numbers, positions, par)
    return h0.build(positions, ovlp, cn=cn)


def overlap(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for overlap integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Overlap integral (matrix) of shape `(nb, nao, nao)`.
    """
    return _integral("_overlap", numbers, positions, par, **kwargs)


def dipole(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for dipole integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Dipole integral (matrix) of shape `(nb, 3, nao, nao)`.
    """
    return _integral("_dipole", numbers, positions, par, **kwargs)


def quadrupole(numbers: Tensor, positions: Tensor, par: Param, **kwargs: Any) -> Tensor:
    """
    Shortcut for dipole integral calculations.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    par : Param
        Representation of an extended tight-binding model.

    Returns
    -------
    Tensor
        Dipole integral (matrix) of shape `(nb, 3, nao, nao)`.
    """
    return _integral("_quadrupole", numbers, positions, par, **kwargs)


def _integral(
    integral_type: Literal["_overlap", "_dipole", "_quadrupole"],
    numbers: Tensor,
    positions: Tensor,
    par: Param,
    **kwargs: Any,
) -> Tensor:
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
