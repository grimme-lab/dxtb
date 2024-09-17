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
Calculators: ABC
================

Abstract helper classes for reduction of boilerplace method definitions (i.e.,
duplication) in calculators.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from dxtb._src.calculators.properties.vibration import (
    IRResult,
    RamanResult,
    VibResult,
)
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Literal, Tensor

__all__ = ["GetPropertiesMixin", "PropertyNotImplementedError"]


class PropertyNotImplementedError(NotImplementedError):
    """Raised if a calculator does not implement the requested property."""


class GetPropertiesMixin(ABC):
    """
    Mixin class which provides :meth:`get_energy`, :meth:`get_forces` and so on.

    Inheriting classes must implement :meth:`get_property`.
    """

    implemented_properties: list[str]
    """Names of implemented methods of the Calculator."""

    def get_implemented_properties(self) -> list[str]:
        return sorted(self.implemented_properties)

    @abstractmethod
    def get_property(
        self,
        name: str,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor | VibResult | IRResult | RamanResult:
        """
        Get the named property.

        Parameters
        ----------
        name : str
            Name of the property to get.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to 0.

        Returns
        -------
        Tensor
            The requested property.
        """

    # energy

    def get_energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "energy", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_potential_energy(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_energy(positions, chrg=chrg, spin=spin, **kwargs)

    # nuclear derivatives

    def get_forces(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        grad_mode: Literal[
            "autograd", "backward", "functorch", "row"
        ] = "autograd",
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Calculate the nuclear forces :math:`f` via AD.

        .. math::

            f = -\dfrac{\partial E}{\partial R}

        One can calculate the Jacobian either row-by-row using the standard
        :func:`torch.autograd.grad` with unit vectors in the VJP or using
        :mod:`torch.func`'s function transforms (e.g.,
        :func:`torch.func.jacrev`).

        Note
        ----
        Using :mod:`torch.func`'s function transforms can apparently be only
        used once. Hence, for example, the Hessian and the dipole derivatives
        cannot be both calculated with functorch.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        grad_mode: Literal, optional
            Specify the mode for gradient calculation. Possible options are:

            - "autograd" (default): Use PyTorch's :func:`torch.autograd.grad`.
            - "backward": Use PyTorch's backward function.
            - "functorch": Use functorch's `jacrev`.
            - "row": Use PyTorch's autograd row-by-row (unnecessary here).

        Returns
        -------
        Tensor
            Atomic forces of shape ``(..., nat, 3)``.
        """
        prop = self.get_property(
            "forces",
            positions,
            chrg=chrg,
            spin=spin,
            grad_mode=grad_mode,
            **kwargs,
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_hessian(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "hessian", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_vibration(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> VibResult:
        prop = self.get_property(
            "vibration", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, VibResult)
        return prop

    def get_normal_modes(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "normal_modes", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_frequencies(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "frequencies", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    # field derivatives

    def get_dipole(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "dipole", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_dipole_moment(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_dipole(positions, chrg=chrg, spin=spin, **kwargs)

    def get_dipole_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "dipole_deriv", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_dipole_derivatives(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_dipole_deriv(positions, chrg=chrg, spin=spin, **kwargs)

    def get_polarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "polarizability", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_pol_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "pol_deriv", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_polarizability_derivatives(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_pol_deriv(positions, chrg=chrg, spin=spin, **kwargs)

    def get_hyperpolarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "hyperpolarizability", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    # spectra

    def get_ir(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> IRResult:
        prop = self.get_property(
            "ir", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, IRResult)
        return prop

    def get_ir_intensities(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "ir", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, IRResult)

        return prop.ints

    def get_raman(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> RamanResult:
        prop = self.get_property(
            "raman", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, RamanResult)
        return prop

    def get_raman_intensities(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "raman_intensity", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, RamanResult)

        return prop.ints

    def get_raman_depol(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "raman_depol", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, RamanResult)

        return prop.depol

    # SCF properties

    def get_bond_orders(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "bond_orders", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_coefficients(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "coefficients", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_density(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "density", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_charges(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        # pylint: disable=import-outside-toplevel
        from dxtb._src.scf.base import Charges

        prop = self.get_property(
            "charges", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Charges)

        return prop.mono

    def get_mulliken_charges(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        return self.get_charges(positions, chrg=chrg, spin=spin, **kwargs)

    def get_iterations(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "iterations", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_mo_energies(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "mo_energies", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_occupation(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        prop = self.get_property(
            "occupation", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Tensor)
        return prop

    def get_potential(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> Tensor:
        # pylint: disable=import-outside-toplevel
        from dxtb._src.scf.base import Potential

        prop = self.get_property(
            "potential", positions, chrg=chrg, spin=spin, **kwargs
        )
        assert isinstance(prop, Potential)
        assert isinstance(prop.mono, Tensor)

        return prop.mono
