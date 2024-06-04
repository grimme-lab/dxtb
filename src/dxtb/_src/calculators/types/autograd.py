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
Calculators: Autograd
=====================

Calculator for the extended tight-binding model with automatic gradients.
"""

from __future__ import annotations

import logging

import torch

from dxtb import OutputHandler, timer
from dxtb._src.components.interactions.field import efield as efield
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Literal, Tensor

from ..properties.vibration import (
    IRResult,
    RamanResult,
    VibResult,
    ir_ints,
    raman_ints_depol,
    vib_analysis,
)
from . import decorators as cdec
from .energy import EnergyCalculator

__all__ = ["AutogradCalculator"]


logger = logging.getLogger(__name__)


class AutogradCalculator(EnergyCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides properties via automatic differentiation.
    """

    implemented_properties = EnergyCalculator.implemented_properties + [
        "forces",
        "hessian",
        "vibration",
        "normal_modes",
        "frequencies",
        #
        "dipole",
        "dipole_deriv",
        "polarizability",
        "pol_deriv",
        "hyperpolarizability",
        "ir",
        "raman",
    ]

    @cdec.requires_positions_grad
    @cdec.cache
    def forces(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        grad_mode: Literal["autograd", "backward", "functorch", "row"] = "autograd",
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
        OutputHandler.write_stdout("\nForces", v=5)
        OutputHandler.write_stdout("------\n", v=5)
        logger.debug("Forces: Starting.")

        # DEVNOTE: We need to pop the `create_graph` and `retain_graph` kwargs
        # to avoid passing them to the energy function, which would add them to
        # the cache key. This would require to pass them to the energy function
        # as well for a cache hit, which is obviously non-sensical.
        kw = {
            "create_graph": kwargs.pop("create_graph", False),
            "retain_graph": kwargs.pop("retain_graph", False),
        }

        if grad_mode == "autograd":
            e = self.energy(positions, chrg, spin, **kwargs)

            timer.start("Forces")
            (deriv,) = torch.autograd.grad(e, positions, **kw)
            timer.stop("Forces")

        elif grad_mode == "backward":
            e = self.energy(positions, chrg, spin, **kwargs)

            timer.start("Forces")
            e.backward(**kw)
            timer.stop("Forces")

            assert positions.grad is not None
            deriv = positions.grad

        elif grad_mode == "functorch":
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            deriv = jacrev(self.energy, argnums=0)(positions, chrg, spin, **kwargs)
            assert isinstance(deriv, Tensor)

        elif grad_mode == "row":
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            energy = self.energy(positions, chrg, spin, **kwargs)

            timer.start("Forces")
            deriv = jac(energy, positions, **kw).reshape(*positions.shape)
            timer.stop("Forces")

        else:
            raise ValueError(f"Unknown grad_mode: {grad_mode}")

        logger.debug(
            "Forces: Re-enforcing contiguous memory layout after "
            "autodiff (grad_mode=%s).",
            grad_mode,
        )

        logger.debug("Forces: All finished.")
        return -deriv

    @cdec.requires_positions_grad
    @cdec.cache
    def hessian(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "forces"] = "forces",
        matrix: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        """
        Calculation of the nuclear Hessian with AD.

        Note
        ----
        The :func:`torch.func.jacrev` function of ``functorch`` requires
        scalars for the expected behavior, i.e., the nuclear Hessian only
        acquires the expected shape of ``(..., nat, 3, nat, 3)`` if the energy
        is provided as a scalar value.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        use_functorch : bool, optional
            Whether to use `functorch` for autodiff. Defaults to ``False``.
        derived_quantity : Literal['energy', 'forces'], optional
            Which derivative to calculate for the Hessian, i.e., derivative of
            forces or energy w.r.t. positions. Defaults to ``'forces'``.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e.,
            ``(nat*3, nat*3)``. Defaults to ``False``.

        Returns
        -------
        Tensor
            Hessian of shape ``(..., nat, 3, nat, 3)`` or
            ``(..., nat*3, nat*3)``.

        Raises
        ------
        RuntimeError
            Positions tensor does not have ``requires_grad=True``.
        """
        logger.debug("Autodiff Hessian: Starting Calculation.")

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            if derived_quantity == "forces":
                hess_func = jacrev(self.forces, argnums=0)
                # specifiy grad_mode here!
                hess = hess_func(positions, chrg, spin, "functorch")

            elif derived_quantity == "energy":
                hess_func = jacrev(jacrev(self.energy, argnums=0), argnums=0)
                hess = hess_func(positions, chrg, spin)

            else:
                raise ValueError(
                    f"Unknown `derived_quantity` '{derived_quantity}'. The "
                    "hessian can be calculated as the derivative of the "
                    "'energy' and the 'forces'."
                )
            assert isinstance(hess, Tensor)
        else:
            # jacrev requires a scalar from `self.energy`!
            if derived_quantity == "forces":
                # pylint: disable=import-outside-toplevel
                from tad_mctc.autograd import jac

                grad_mode = kwargs.pop("grad_mode", "autograd")
                forces = self.forces(
                    positions,
                    chrg,
                    spin,
                    grad_mode=grad_mode,
                    create_graph=True,
                    retain_graph=True,
                )

                # reshape (..., nat, 3, nat*3) to (..., nat, 3, nat, 3)
                hess = jac(forces, positions).reshape(
                    [*self.numbers.shape[:-1], *2 * [self.numbers.shape[-1], 3]]
                )

            elif derived_quantity == "energy":
                # pylint: disable=import-outside-toplevel
                from tad_mctc.autograd import hessian

                hess = hessian(self.energy, (positions, chrg, spin), argnums=0)

            else:
                raise ValueError(
                    f"Unknown `derived_quantity` '{derived_quantity}'. The "
                    "hessian can be calculated as the derivative of the "
                    "'energy' and the 'forces'."
                )

        if hess.is_contiguous() is False:
            logger.debug(
                "Hessian: Re-enforcing contiguous memory layout after "
                "autodiff (use_functorch=%s).",
                use_functorch,
            )
            hess = hess.contiguous()

        # forces are negative gradient -> revert sign again
        if derived_quantity == "forces":
            hess = -hess

        # reshape (..., nat, 3, nat, 3) to (..., nat*3, nat*3)
        if matrix is True:
            s = [*self.numbers.shape[:-1], *2 * [3 * self.numbers.shape[-1]]]
            hess = hess.view(*s)

        logger.debug("Autodiff Hessian: All finished.")

        return hess

    @cdec.cache
    def vibration(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        project_translational: bool = True,
        project_rotational: bool = True,
        **kwargs: Any,
    ) -> VibResult:
        r"""
        Perform vibrational analysis. This calculates the Hessian matrix and
        diagonalizes it to obtain the vibrational frequencies and normal modes.

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
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        project_translational : bool, optional
            Project out translational modes. Defaults to ``True``.
        project_rotational : bool, optional
            Project out rotational modes. Defaults to ``True``.

        Returns
        -------
        VibResult
            Result container with vibrational frequencies (shape:
            ``(..., nfreqs)``) and normal modes (shape:
            ``(..., nat*3, nfreqs)``).
        """
        hess = self.hessian(
            positions,
            chrg,
            spin,
            use_functorch=use_functorch,
            matrix=False,
            **kwargs,
        )
        a = vib_analysis(
            self.numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
            **kwargs,
        )

        return a

    @cdec.requires_efield
    @cdec.requires_efield_grad
    @cdec.cache
    def dipole(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate the electric dipole moment :math:`\mu` via AD.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

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
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Electric dipole moment of shape ``(..., 3)``.
        """
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(positions, chrg, spin)

            dip = jacrev(wrapped_energy)(field)
            assert isinstance(dip, Tensor)
        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            # calculate electric dipole contribution from xtb energy: -de/dE
            energy = self.energy(positions, chrg, spin)
            dip = jac(energy, field)

        if dip.is_contiguous() is False:
            logger.debug(
                "Dipole moment: Re-enforcing contiguous memory layout "
                "after autodiff (use_functorch=%s).",
                use_functorch,
            )
            dip = dip.contiguous()

        return -dip

    @cdec.requires_positions_grad
    @cdec.cache
    def dipole_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_analytical: bool = True,
        use_functorch: bool = False,
    ) -> Tensor:
        r"""
        Calculate cartesian dipole derivative :math:`\mu'`.

        .. math::

            \mu' = \dfrac{\partial \mu}{\partial R} = \dfrac{\partial^2 E}{\partial F \partial R}

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
        use_analytical: bool, optional
            Whether to use the analytically calculated dipole moment for AD or
            the automatically differentiated dipole moment.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape ``(..., 3, nat, 3)``.
        """

        if use_analytical is True:
            if not hasattr(self, "dipole_analytical") or not callable(
                getattr(self, "dipole_analytical")
            ):
                raise ValueError(
                    "Analytical dipole moment not available. "
                    "Please use a calculator, which subclasses "
                    "the `AnalyticalCalculator`."
                )
            dip_fcn = self.dipole_analytical  # type: ignore
        else:
            dip_fcn = self.dipole

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # d(3) / d(nat, 3) = (3, nat, 3)
            dmu_dr = jacrev(dip_fcn, argnums=0)(positions, chrg, spin, use_functorch)
            assert isinstance(dmu_dr, Tensor)

        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            mu = dip_fcn(positions, chrg, spin, use_functorch)

            # (..., 3, 3*nat) -> (..., 3, nat, 3)
            dmu_dr = jac(mu, positions).reshape(
                (*self.numbers.shape[:-1], 3, *positions.shape[-2:])
            )

        if dmu_dr.is_contiguous() is False:
            logger.debug(
                "Dipole derivative: Re-enforcing contiguous memory layout "
                "after autodiff (use_functorch=%s).",
                use_functorch,
            )
            dmu_dr = dmu_dr.contiguous()

        return dmu_dr

    @cdec.requires_efield
    @cdec.requires_efield_grad
    @cdec.cache
    def polarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        use_analytical: bool = False,
        derived_quantity: Literal["energy", "dipole"] = "dipole",
    ) -> Tensor:
        r"""
        Calculate the polarizability tensor :math:`\alpha`.

        .. math::

            \alpha = \dfrac{\partial \mu}{\partial F} = \dfrac{\partial^2 E}{\partial^2 F}

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
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Polarizability tensor of shape ``(..., 3, 3)``.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_analytical is True:
            if not hasattr(self, "dipole_analytical") or not callable(
                getattr(self, "dipole_analytical")
            ):
                raise ValueError(
                    "Analytical dipole moment not available. "
                    "Please use a calculator, which subclasses "
                    "the `AnalyticalCalculator`."
                )

            # FIXME: Not working for Raman
            dip_fcn = self.dipole_analytical  # type: ignore
        else:
            dip_fcn = self.dipole

        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            mu = dip_fcn(positions, chrg, spin)
            return jac(mu, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return dip_fcn(positions, chrg, spin)

            alpha = jacrev(wrapped_dipole)(field)
            assert isinstance(alpha, Tensor)
        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(positions, chrg, spin)

            alpha = jacrev(jacrev(wrapped_energy))(field)
            assert isinstance(alpha, Tensor)

            # negative sign already in dipole but not in energy derivative!
            alpha = -alpha
        else:
            raise ValueError(
                f"Unknown `derived_quantity` '{derived_quantity}'. The "
                "polarizability can be calculated as the derivative of the "
                "'dipole' moment or the 'energy'."
            )

        if alpha.is_contiguous() is False:
            logger.debug(
                "Polarizability: Re-enforcing contiguous memory layout "
                "after autodiff (use_functorch=%s).",
                use_functorch,
            )
            alpha = alpha.contiguous()

        # 3x3 polarizability tensor
        return alpha

    @cdec.requires_efield
    @cdec.requires_positions_grad
    @cdec.cache
    def pol_deriv(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole"] = "dipole",
    ) -> Tensor:
        r"""
        Calculate the cartesian polarizability derivative :math:`\chi`.

        .. math::

            \chi = \alpha'
            = \dfrac{\partial \alpha}{\partial R}
            = \dfrac{\partial^3 E}{\partial^2 F \partial R}

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
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Polarizability derivative shape ``(..., 3, 3, nat, 3)``.
        """
        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            a = self.polarizability(positions, chrg, spin, use_functorch=use_functorch)

            # d(3, 3) / d(nat, 3) -> (3, 3, nat*3) -> (3, 3, nat, 3)
            chi = jac(a, positions).reshape((3, 3, *positions.shape[-2:]))

        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            chi = jacrev(self.polarizability, argnums=0)(
                positions, chrg, spin, use_functorch, derived_quantity
            )
            assert isinstance(chi, Tensor)

        if chi.is_contiguous() is False:
            logger.debug(
                "Polarizability derivative: Re-enforcing contiguous memory "
                "layout after autodiff (use_functorch=%s).",
                use_functorch,
            )
            chi = chi.contiguous()

        return chi

    @cdec.requires_efield
    @cdec.requires_efield_grad
    @cdec.cache
    def hyperpolarizability(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole", "polarizability", "pol"] = "pol",
    ) -> Tensor:
        r"""
        Calculate the hyper polarizability tensor :math:`\beta`.

        .. math::

            \beta = \dfrac{\partial \alpha}{\partial F}
            = \dfrac{\partial^2 \mu}{\partial F^2}
            = \dfrac{\partial^3 E}{\partial^2 3}

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
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Hyper polarizability tensor of shape ``(..., 3, 3, 3)``.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            alpha = self.polarizability(
                positions, chrg, spin, use_functorch=use_functorch
            )
            return jac(alpha, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "pol":

            def wrapped_polarizability(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.polarizability(positions, chrg, spin)

            beta = jacrev(wrapped_polarizability)(field)

        elif derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.dipole(positions, chrg, spin)

            beta = jacrev(jacrev(wrapped_dipole))(field)

        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(positions, chrg, spin)

            beta = jacrev(jacrev(jacrev(wrapped_energy)))(field)

        else:
            raise ValueError(
                f"Unknown `derived_quantity` '{derived_quantity}'. The "
                "polarizability can be calculated as the derivative of the "
                "'dipole' moment or the 'energy'."
            )

        # 3x3x3 hyper polarizability tensor
        assert isinstance(beta, Tensor)

        # usually only after jacrev not contiguous
        if beta.is_contiguous() is False:
            logger.debug(
                "Hyperpolarizability: Re-enforcing contiguous memory "
                "layout after autodiff (use_functorch=%s).",
                use_functorch,
            )
            beta = beta.contiguous()

        if derived_quantity == "energy":
            beta = -beta
        return beta

    # SPECTRA

    @cdec.cache
    def ir(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> IRResult:
        """
        Calculate the frequencies and intensities of IR spectra.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to ``False``.

        Returns
        -------
        IRResult
            Result container with frequencies (shape: ``(..., nfreqs)``) and
            intensities (shape: ``(..., nfreqs)``) of IR spectra.
        """
        OutputHandler.write_stdout("\nIR Spectrum")
        OutputHandler.write_stdout("-----------")
        logger.debug("IR spectrum: Start.")

        # run vibrational analysis first
        vib_res = self.vibration(positions, chrg, spin)

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.classicals.reset_all()
        self.interactions.reset_all()
        self.integrals.reset_all()

        # calculate nuclear dipole derivative dmu/dR: (..., 3, nat, 3)
        dmu_dr = self.dipole_deriv(positions, chrg, spin, use_functorch=use_functorch)

        intensities = ir_ints(dmu_dr, vib_res.modes)

        logger.debug("IR spectrum: All finished.")

        return IRResult(vib_res.freqs, intensities)

    @cdec.cache
    def raman(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> RamanResult:
        """
        Calculate the frequencies, static intensities and depolarization ratio
        of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to ``False``.

        Returns
        -------
        RamanResult
            Result container with frequencies (shape: ``(..., nfreqs)``),
            intensities (shape: ``(..., nfreqs)``) and the depolarization ratio
            (shape: ``(..., nfreqs)``) of Raman spectra.
        """
        OutputHandler.write_stdout("\nRaman Spectrum")
        OutputHandler.write_stdout("--------------")
        logger.debug("Raman spectrum: Start.")

        vib_res = self.vibration(positions, chrg, spin, use_functorch=use_functorch)

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.reset()

        # d(..., 3, 3) / d(..., nat, 3) -> (..., 3, 3, nat, 3)
        da_dr = self.pol_deriv(positions, chrg, spin, use_functorch=use_functorch)

        intensities, depol = raman_ints_depol(da_dr, vib_res.modes)

        logger.debug("Raman spectrum: All finished.")

        return RamanResult(vib_res.freqs, intensities, depol)

    def calculate(
        self,
        properties: list[str],
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        **kwargs: Any,
    ) -> None:
        """
        Calculate the requested properties. This is more of a dispatcher method
        that calls the appropriate methods of the Calculator.

        Parameters
        ----------
        properties : list[str]
            List of properties to calculate.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        """
        # DEVNOTE: `super()` does not quite work, because the default kwargs of
        # other functions may be missing. Example: Running `forces` will always
        # use the `grad_mode` argument, which is not known within `super()`.
        # Hence, the cache key will be different and the energy calculation
        # after `forces` would not use the cached energy.
        # super().calculate(properties, positions, chrg, spin, **kwargs)

        if self.opts.cache.enabled is False:
            self.cache.reset_all()

        # treat bond orders separately for better error message
        if "bond_orders" in properties:
            self.bond_orders(positions, chrg, spin, **kwargs)

        props = list(EnergyCalculator.implemented_properties)
        props.remove("bond_orders")
        if set(props) & set(properties):
            self.energy(positions, chrg, spin, **kwargs)

        if "forces" in properties:
            self.forces(positions, chrg, spin, **kwargs)

        if "hessian" in properties:
            self.hessian(positions, chrg, spin, **kwargs)

        if {"vibration", "frequencies", "normal_modes"} & set(properties):
            self.vibration(positions, chrg, spin, **kwargs)

        if "dipole" in properties:
            self.dipole(positions, chrg, spin, **kwargs)

        if {"dipole_derivatives", "dipole_deriv"} & set(properties):
            self.dipole_deriv(positions, chrg, spin, **kwargs)

        if "polarizability" in properties:
            self.polarizability(positions, chrg, spin, **kwargs)

        if {"polarizability_derivatives", "pol_deriv"} & set(properties):
            self.pol_deriv(positions, chrg, spin, **kwargs)

        if "hyperpolarizability" in properties:
            self.hyperpolarizability(positions, chrg, spin, **kwargs)

        if {"ir", "ir_intensities"} in set(properties):
            self.ir(positions, chrg, spin, **kwargs)

        if {"raman", "raman_intensities", "raman_depol"} & set(properties):
            self.raman(positions, chrg, spin, **kwargs)
