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

from dxtb.components.interactions.field import efield as efield
from dxtb.constants import defaults
from dxtb.io import OutputHandler
from dxtb.properties import vibration as vib
from dxtb.typing import Any, Literal, Tensor

from . import decorators as cdec
from .energy import EnergyCalculator

__all__ = ["AutogradCalculator"]


logger = logging.getLogger(__name__)


class AutogradCalculator(EnergyCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides properties via automatic differentiation.
    """

    @cdec.requires_positions_grad
    @cdec.cache
    def forces(
        self,
        numbers: Tensor,
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
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the dipole derivatives cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        grad_mode: Literal, optional
            Specify the mode for gradient calculation. Possible options are:
            - "autograd" (default): Use PyTorch's `torch.autograd.grad`.
            - "backward": Use PyTorch's backward function.
            - "functorch": Use functorch's `jacrev`.
            - "row": Use PyTorch's autograd row-by-row (unnecessary here).

        Returns
        -------
        Tensor
            Atomic forces of shape `(..., nat, 3)`.
        """
        OutputHandler.write_stdout("\nForces", v=5)
        OutputHandler.write_stdout("------\n", v=5)
        logger.debug("Forces: Starting.")

        kw = {
            "create_graph": kwargs.pop("create_graph", False),
            "retain_graph": kwargs.pop("retain_graph", False),
        }

        if grad_mode == "autograd":
            e = self.energy(numbers, positions, chrg, spin)
            (deriv,) = torch.autograd.grad(e, positions, **kw)

        elif grad_mode == "backward":
            e = self.energy(numbers, positions, chrg, spin)
            e.backward(**kw)
            assert positions.grad is not None
            deriv = positions.grad

        elif grad_mode == "functorch":
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            jac_func = jacrev(self.energy, argnums=1)
            deriv = jac_func(numbers, positions, chrg, spin)
            assert isinstance(deriv, Tensor)

        elif grad_mode == "row":
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            energy = self.energy(numbers, positions, chrg, spin)
            deriv = jac(energy, positions, **kw).reshape(*positions.shape)

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
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        matrix: bool = False,
    ) -> Tensor:
        """
        Calculation of the nuclear Hessian with AD.

        Note
        ----
        The `jacrev` function of `functorch` requires scalars for the expected
        behavior, i.e., the nuclear Hessian only acquires the expected shape of
        `(nat, 3, nat, 3)` if the energy is provided as a scalar value.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use `functorch` for autodiff. Defaults to `False`.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e., `(nat*3, nat*3)`.
            Defaults to `False`.

        Returns
        -------
        Tensor
            Hessian of shape `(..., nat, 3, nat, 3)` or `(..., nat*3, nat*3)`.

        Raises
        ------
        RuntimeError
            Positions tensor does not have `requires_grad=True`.
        """
        logger.debug("Autodiff Hessian: Starting Calculation.")

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # jacrev requires a scalar from `self.energy`!
            hess_func = jacrev(jacrev(self.energy, argnums=1), argnums=1)
            hess = hess_func(numbers, positions, chrg, spin)
            assert isinstance(hess, Tensor)
        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import hessian

            # jacrev requires a scalar from `self.energy`!
            hess = hessian(self.energy, (numbers, positions, chrg, spin), argnums=1)

        if hess.is_contiguous() is False:
            logger.debug(
                "Hessian: Re-enforcing contiguous memory layout after "
                "autodiff (use_functorch=%s).",
                use_functorch,
            )
            hess = hess.contiguous()

        # reshape (..., nat, 3, nat, 3) to (..., nat*3, nat*3)
        if matrix is True:
            s = [*numbers.shape[:-1], *2 * [3 * numbers.shape[-1]]]
            hess = hess.view(*s)

        self.cache["hessian"] = hess

        logger.debug("Autodiff Hessian: All finished.")

        return hess

    def vibration(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> vib.VibResult:
        r"""
        Perform vibrational analysis. This calculates the Hessian matrix and
        diagonalizes it to obtain the vibrational frequencies and normal modes.

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the dipole derivatives cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        project_translational : bool, optional
            Project out translational modes. Defaults to `True`.
        project_rotational : bool, optional
            Project out rotational modes. Defaults to `True`.

        Returns
        -------
        vib.VibResult
            Result container with vibrational frequencies (shape:
            `(..., nfreqs)`) and normal modes (shape: `(..., nat*3, nfreqs)`).
        """
        hess = self.hessian(
            numbers,
            positions,
            chrg,
            spin,
            use_functorch=use_functorch,
            matrix=False,
        )
        return vib.vib_analysis(
            numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    @cdec.requires_efield
    @cdec.requires_efield_grad
    def dipole(
        self,
        numbers: Tensor,
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
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the dipole derivatives cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

            dip = jacrev(wrapped_energy)(field)
            assert isinstance(dip, Tensor)
        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            # calculate electric dipole contribution from xtb energy: -de/dE
            energy = self.energy(numbers, positions, chrg, spin)
            dip = jac(energy, field)

        if dip.is_contiguous() is False:
            logger.debug(
                "Dipole moment: Re-enforcing contiguous memory layout "
                "after autodiff (use_functorch=%s).",
                use_functorch,
            )
            dip = dip.contiguous()

        return -dip

    @cdec.requires_efield
    def dipole_analytical(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        *_,  # absorb stuff
    ) -> Tensor:
        r"""
        Analytically calculate the electric dipole moment :math:`\mu`.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.

        Returns
        -------
        Tensor
            Electric dipole moment of shape `(..., 3)`.
        """
        # run single point and check if integral is populated
        result = self.singlepoint(numbers, positions, chrg, spin)

        dipint = self.integrals.dipole
        if dipint is None:
            raise RuntimeError(
                "Dipole moment requires a dipole integral. They should "
                f"be added automatically if the '{efield.LABEL_EFIELD}' "
                "interaction is added to the Calculator."
            )
        if dipint.matrix is None:
            raise RuntimeError(
                "Dipole moment requires a dipole integral. They should "
                f"be added automatically if the '{efield.LABEL_EFIELD}' "
                "interaction is added to the Calculator."
            )

        # pylint: disable=import-outside-toplevel
        from dxtb.properties.moments.dip import dipole

        # dip = properties.dipole(
        # numbers, positions, result.density, self.integrals.dipole
        # )
        qat = self.ihelp.reduce_orbital_to_atom(result.charges.mono)
        dip = dipole(qat, positions, result.density, dipint.matrix)
        return dip

    @cdec.requires_positions_grad
    def dipole_deriv(
        self,
        numbers: Tensor,
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
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the dipole derivatives cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_analytical: bool, optional
            Whether to use the analytically calculated dipole moment for AD or
            the automatically differentiated dipole moment.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape `(..., 3, nat, 3)`.
        """

        if use_analytical is True:
            dip_fcn = self.dipole_analytical
        else:
            dip_fcn = self.dipole

        if use_functorch is True:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            # d(3) / d(nat, 3) = (3, nat, 3)
            dmu_dr = jacrev(dip_fcn, argnums=1)(
                numbers, positions, chrg, spin, use_functorch
            )
            assert isinstance(dmu_dr, Tensor)

        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            mu = dip_fcn(numbers, positions, chrg, spin, use_functorch)

            # (..., 3, 3*nat) -> (..., 3, nat, 3)
            dmu_dr = jac(mu, positions).reshape(
                (*numbers.shape[:-1], 3, *positions.shape[-2:])
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
    def polarizability(
        self,
        numbers: Tensor,
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
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Polarizability tensor of shape `(..., 3, 3)`.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_analytical is True:
            # FIXME: Not working for Raman
            dip_fcn = self.dipole_analytical
        else:
            dip_fcn = self.dipole

        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            mu = dip_fcn(numbers, positions, chrg, spin)
            return jac(mu, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return dip_fcn(numbers, positions, chrg, spin)

            alpha = jacrev(wrapped_dipole)(field)
            assert isinstance(alpha, Tensor)
        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

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
    def pol_deriv(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole"] = "dipole",
    ) -> Tensor:
        r"""
        Calculate the cartesian polarizability derivative :math:`\chi`.

        .. math::

            \begin{align*}
                \chi &= \alpha' \\
                    &= \dfrac{\partial \alpha}{\partial R} \\
                    &= \dfrac{\partial^2 \mu}{\partial F \partial R} \\
                    &= \dfrac{\partial^3 E}{\partial^2 F \partial R}
            \end{align*}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system of shape `(..., nat)`.
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Polarizability derivative shape `(..., 3, 3, nat, 3)`.
        """
        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            a = self.polarizability(
                numbers, positions, chrg, spin, use_functorch=use_functorch
            )

            # d(3, 3) / d(nat, 3) -> (3, 3, nat*3) -> (3, 3, nat, 3)
            chi = jac(a, positions).reshape((3, 3, *positions.shape[-2:]))

        else:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jacrev

            chi = jacrev(self.polarizability, argnums=1)(
                numbers, positions, chrg, spin, use_functorch, derived_quantity
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
    def hyperpol(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
        derived_quantity: Literal["energy", "dipole", "polarizability", "pol"] = "pol",
    ) -> Tensor:
        r"""
        Calculate the hyper polarizability tensor :math:`\beta`.

        .. math::

            \begin{align*}
                \beta &= \dfrac{\partial \alpha}{\partial F} \\
                    &= \dfrac{\partial^2 \mu}{\partial F^2}
                    &= \dfrac{\partial^3 E}{\partial^2 3} \\
            \end{align*}

        One can calculate the Jacobian either row-by-row using the standard
        `torch.autograd.grad` with unit vectors in the VJP (see `here`_) or
        using `torch.func`'s function transforms (e.g., `jacrev`).

        .. _here: https://pytorch.org/functorch/stable/notebooks/\
                  jacobians_hessians.html#computing-the-jacobian

        Note
        ----
        Using `torch.func`'s function transforms can apparently be only used
        once. Hence, for example, the Hessian and the polarizability cannot
        be both calculated with functorch.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch: bool, optional
            Whether to use functorch or the standard (slower) autograd.
        derived_quantity: Literal['energy', 'dipole'], optional
            Which derivative to calculate for the polarizability, i.e.,
            derivative of dipole moment or energy w.r.t field.

        Returns
        -------
        Tensor
            Hyper polarizability tensor of shape `(..., 3, 3, 3)`.
        """
        # retrieve the efield interaction and the field
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        if use_functorch is False:
            # pylint: disable=import-outside-toplevel
            from tad_mctc.autograd import jac

            alpha = self.polarizability(
                numbers, positions, chrg, spin, use_functorch=use_functorch
            )
            return jac(alpha, field)

        # pylint: disable=import-outside-toplevel
        from tad_mctc.autograd import jacrev

        if derived_quantity == "pol":

            def wrapped_polarizability(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.polarizability(numbers, positions, chrg, spin)

            beta = jacrev(wrapped_polarizability)(field)

        elif derived_quantity == "dipole":

            def wrapped_dipole(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.dipole(numbers, positions, chrg, spin)

            beta = jacrev(jacrev(wrapped_dipole))(field)

        elif derived_quantity == "energy":

            def wrapped_energy(f: Tensor) -> Tensor:
                self.interactions.update_efield(field=f)
                return self.energy(numbers, positions, chrg, spin)

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

    def ir(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> vib.IRResult:
        """
        Calculate the frequencies and intensities of IR spectra.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to `False`.

        Returns
        -------
        vib.IRResult
            Result container with frequencies (shape: `(..., nfreqs)`) and
            intensities (shape: `(..., nfreqs)`) of IR spectra.
        """
        OutputHandler.write_stdout("\nIR Spectrum")
        OutputHandler.write_stdout("-----------")
        logger.debug("IR spectrum: Start.")

        # run vibrational analysis first
        vib_res = self.vibration(numbers, positions, chrg, spin)

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.integrals.reset_all()

        # calculate nuclear dipole derivative dmu/dR: (..., 3, nat, 3)
        dmu_dr = self.dipole_deriv(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        intensities = vib.ir_ints(dmu_dr, vib_res.modes)

        logger.debug("IR spectrum: All finished.")

        return vib.IRResult(vib_res.freqs, intensities)

    def raman(
        self,
        numbers: Tensor,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        use_functorch: bool = False,
    ) -> vib.RamanResult:
        """
        Calculate the frequencies, static intensities and depolarization ratio
        of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers for all atoms in the system (shape: `(..., nat)`).
        positions : Tensor
            Cartesian coordinates of all atoms (shape: `(..., nat, 3)`).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to `None`.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to `None`.
        use_functorch : bool, optional
            Whether to use functorch or the standard (slower) autograd.
            Defaults to `False`.

        Returns
        -------
        vib.RamanResult
            Result container with frequencies (shape: `(..., nfreqs)`),
            intensities (shape: `(..., nfreqs)`) and the depolarization ratio
            (shape: `(..., nfreqs)`) of Raman spectra.
        """
        OutputHandler.write_stdout("\nRaman Spectrum")
        OutputHandler.write_stdout("--------------")
        logger.debug("Raman spectrum: Start.")

        vib_res = self.vibration(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        # TODO: Figure out how to run func transforms 2x properly
        # (improve: Hessian does not need dipole integral but dipder does)
        self.integrals.reset_all()

        # d(..., 3, 3) / d(..., nat, 3) -> (..., 3, 3, nat, 3)
        da_dr = self.pol_deriv(
            numbers, positions, chrg, spin, use_functorch=use_functorch
        )

        intensities, depol = vib.raman_ints_depol(da_dr, vib_res.modes)

        logger.debug("Raman spectrum: All finished.")

        return vib.RamanResult(vib_res.freqs, intensities, depol)
