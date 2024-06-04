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
Calculators: Numerical
======================

Calculator for the extended tight-binding model with numerical differentiation.
"""

from __future__ import annotations

import logging

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions.field import efield
from dxtb._src.constants import defaults
from dxtb._src.typing import Any, Tensor

from ..properties import vibration as vib
from . import decorators as cdec
from .energy import EnergyCalculator

__all__ = ["NumericalCalculator"]


logger = logging.getLogger(__name__)


class NumericalCalculator(EnergyCalculator):
    """
    Parametrized calculator defining the extended tight-binding model.

    This class provides various molecular properties through numerical
    differentiation.
    """

    implemented_properties = EnergyCalculator.implemented_properties + [
        "forces",
        "hessian",
        "normal_modes",
        "frequencies",
        "dipole",
        "dipole_deriv",
        "polarizability",
        "pol_deriv",
        "hyperpolarizability",
        "ir",
        "raman",
    ]
    """Names of implemented methods of the Calculator."""

    @cdec.numerical
    @cdec.cache
    def forces_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
        **kwargs: Any,
    ) -> Tensor:
        r"""
        Numerically calculate the atomic forces :math:`f`.

        .. math::

            f = -\dfrac{\partial E}{\partial R}

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Atomic forces of shape ``(..., nat, 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # (..., nat, 3)
        deriv = torch.zeros(positions.shape, **self.dd)
        logger.debug("Forces (numerical): Starting build (%s).", deriv.shape)

        OutputHandler.write_stdout("Forces (numerical)\n", v=4)

        linebreak = kwargs.pop("linebreak", 20)
        nsteps = 3 * self.numbers.shape[-1]
        count = 1

        OutputHandler.write_stdout(
            f"Starting build of force matrix: {deriv.shape[-2]} x "
            f"{deriv.shape[-1]} ({nsteps} steps, {nsteps*2} evaluations)",
            v=4,
        )

        for i in range(self.numbers.shape[-1]):
            for j in range(3):
                with OutputHandler.with_verbosity(0):
                    positions[..., i, j] += step_size
                    gr = self.energy(positions, chrg, spin)

                    positions[..., i, j] -= 2 * step_size
                    gl = self.energy(positions, chrg, spin)

                    positions[..., i, j] += step_size
                    deriv[..., i, j] = 0.5 * (gr - gl) / step_size

                if count % linebreak == 0:
                    OutputHandler.write_stdout(f". {count}/{nsteps}", v=4)
                else:
                    OutputHandler.write_stdout_nf(".", v=4)

                logger.debug("Forces (numerical): step %s/%s", count, nsteps)
                count += 1

                gc.collect()
            gc.collect()

        # set counter to correct value
        count -= 1

        logger.debug("Forces (numerical): All finished.")
        space = (linebreak - count % linebreak) * " "
        OutputHandler.write_stdout(f"{space} {count}/{nsteps}", v=4)

        return -deriv

    @cdec.numerical
    @cdec.cache
    def hessian_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
        matrix: bool = False,
    ) -> Tensor:
        """
        Numerically calculate the Hessian.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : int | float, optional
            Step size for numerical differentiation.
        matrix : bool, optional
            Whether to reshape the Hessian to a matrix, i.e.,
            ``(nat*3, nat*3)``. Defaults to ``False``.

        Returns
        -------
        Tensor
            Hessian of shape ``(..., nat, 3, nat, 3)`` or
            ``(..., nat*3, nat*3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # potentially use analytical forces if available
        if hasattr(self, "forces_analytical") and callable(
            getattr(self, "forces_analytical")
        ):

            def _gradfcn(pos: Tensor) -> Tensor:
                with torch.enable_grad():
                    pos.requires_grad_(True)
                    result = -self.forces_analytical(pos, chrg, spin)  # type: ignore
                    pos.detach_()
                return result.detach()

        else:

            def _gradfcn(pos: Tensor) -> Tensor:
                return -self.forces_numerical(pos, chrg, spin)

        # (..., nat, 3, nat, 3)
        deriv = torch.zeros((*positions.shape, *positions.shape[-2:]), **self.dd)
        logger.debug("Hessian (numerical): Starting build (%s).", deriv.shape)

        count = 1
        nsteps = 3 * self.numbers.shape[-1]
        for i in range(self.numbers.shape[-1]):
            for j in range(3):
                with OutputHandler.with_verbosity(0):
                    positions[..., i, j] += step_size
                    gr = _gradfcn(positions)

                    positions[..., i, j] -= 2 * step_size
                    gl = _gradfcn(positions)

                    positions[..., i, j] += step_size
                    deriv[..., :, :, i, j] = 0.5 * (gr - gl) / step_size

                logger.debug("Hessian (numerical): step %s/%s", count, nsteps)
                count += 1

                gc.collect()
            gc.collect()

        # reshape (..., nat, 3, nat, 3) to (..., nat*3, nat*3)
        if matrix is True:
            s = [*self.numbers.shape[:-1], *2 * [3 * self.numbers.shape[-1]]]
            deriv = deriv.reshape(*s)

        logger.debug("Hessian (numerical): All finished.")

        return deriv

    @cdec.numerical
    @cdec.cache
    def vibration_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
        project_translational: bool = True,
        project_rotational: bool = True,
    ) -> vib.VibResult:
        r"""
        Perform vibrational analysis via numerical Hessian.
        The Hessian matrix is diagonalized to obtain the vibrational
        frequencies and normal modes.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : int | float, optional
            Step size for numerical differentiation.
        project_translational : bool, optional
            Project out translational modes. Defaults to ``True``.
        project_rotational : bool, optional
            Project out rotational modes. Defaults to ``True``.

        Returns
        -------
        vib.VibResult
            Result container with vibrational frequencies (shape:
            ``(..., nfreqs)``) and normal modes (shape:
            ``(..., nat*3, nfreqs)``).
        """
        hess = self.hessian_numerical(positions, chrg, spin, step_size=step_size)
        return vib.vib_analysis(
            self.numbers,
            positions,
            hess,
            project_translational=project_translational,
            project_rotational=project_rotational,
        )

    # PROPERTIES (FIELD)

    @cdec.numerical
    @cdec.requires_efield
    @cdec.cache
    def dipole_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the electric dipole moment :math:`\mu`.

        .. math::

            \mu = \dfrac{\partial E}{\partial F}

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Electric dipole moment of shape ``(..., 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # retrieve electric field, no copy needed because of no_grad context
        field = self.interactions.get_interaction(efield.LABEL_EFIELD).field

        # (..., 3)
        deriv = torch.zeros((*self.numbers.shape[:-1], 3), **self.dd)
        logger.debug("Dipole (numerical): Starting build (%s).", deriv.shape)

        count = 1
        for i in range(3):
            with OutputHandler.with_verbosity(0):
                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                gr = self.energy(positions, chrg, spin)

                field[..., i] -= 2 * step_size
                self.interactions.update_efield(field=field)
                gl = self.energy(positions, chrg, spin)

                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                deriv[..., i] = 0.5 * (gr - gl) / step_size

            logger.debug("Dipole (numerical): step %s/3.", count)
            count += 1

            gc.collect()

        logger.debug("Dipole (numerical): All finished.")

        return -deriv

    @cdec.numerical
    @cdec.cache
    def dipole_deriv_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate cartesian dipole derivative :math:`\mu'`.

        .. math::

            \mu' = \dfrac{\partial \mu}{\partial R}
                 = \dfrac{\partial^2 E}{\partial F \partial R}

        Here, the analytical dipole moment is used for the numerical
        differentiation (if it is available in the calculator).

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size: int | float, optional
            Step size for numerical differentiation.

        Returns
        -------
        Tensor
            Cartesian dipole derivative of shape ``(..., 3, nat, 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        if hasattr(self, "dipole_analytical") and callable(
            getattr(self, "dipole_analytical")
        ):
            _dipfcn = self.dipole_analytical  # type: ignore
        else:
            _dipfcn = self.dipole_numerical

        # (..., 3, n, 3)
        deriv = torch.zeros(
            (*self.numbers.shape[:-1], 3, *positions.shape[-2:]),
            **self.dd,
        )
        logger.debug("Dipole derivative (numerical): Starting build (%s).", deriv.shape)

        count = 1
        nsteps = 3 * self.numbers.shape[-1]

        for i in range(self.numbers.shape[-1]):
            for j in range(3):
                with OutputHandler.with_verbosity(0):
                    positions[..., i, j] += step_size
                    r = _dipfcn(positions, chrg, spin)

                    positions[..., i, j] -= 2 * step_size
                    l = _dipfcn(positions, chrg, spin)

                    positions[..., i, j] += step_size
                    deriv[..., :, i, j] = 0.5 * (r - l) / step_size

                logger.debug("Dipole derivative (numerical): Step %s/%s", count, nsteps)
                count += 1

                gc.collect()
            gc.collect()

        logger.debug("Dipole derivative (numerical): All finished.")

        return deriv

    @cdec.numerical
    @cdec.requires_efield
    @cdec.cache
    def polarizability_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the polarizability tensor :math:`\alpha`.

        .. math::

            \alpha = \dfrac{\partial \mu}{\partial F} = \dfrac{\partial^2 E}{\partial^2 F}

        Here, the analytical dipole moment is used for the numerical derivative.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability tensor of shape ``(..., 3, 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        if hasattr(self, "dipole_analytical") and callable(
            getattr(self, "dipole_analytical")
        ):
            _dipfcn = self.dipole_analytical  # type: ignore
        else:
            _dipfcn = self.dipole_numerical

        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        _field = ef.field.clone()
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # (..., 3, 3)
        deriv = torch.zeros(*(*self.numbers.shape[:-1], 3, 3), **self.dd)
        logger.debug("Polarizability (numerical): Starting build %s", deriv.shape)

        count = 1
        for i in range(3):
            with OutputHandler.with_verbosity(0):
                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                gr = _dipfcn(positions, chrg, spin)

                field[..., i] -= 2 * step_size
                self.interactions.update_efield(field=field)
                gl = _dipfcn(positions, chrg, spin)

                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                deriv[..., :, i] = 0.5 * (gr - gl) / step_size

            logger.debug("Polarizability (numerical): step %s/3", count)
            count += 1

            gc.collect()

        logger.debug("Polarizability (numerical): All finished.")

        # explicitly update field (to restore original field with possible grad)
        self.interactions.reset_efield()
        self.interactions.update_efield(field=_field)

        return deriv

    @cdec.numerical
    @cdec.requires_efield
    @cdec.cache
    def pol_deriv_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the cartesian polarizability derivative
        :math:`\chi`.

        .. math::

            \chi = \alpha'
            = \dfrac{\partial \alpha}{\partial R}
            = \dfrac{\partial^2 \mu}{\partial F \partial R}
            = \dfrac{\partial^3 E}{\partial^2 F \partial R}

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Polarizability derivative shape ``(..., 3, 3, nat, 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # (..., 3, 3, nat, 3)
        deriv = torch.zeros(
            (*self.numbers.shape[:-1], 3, 3, *positions.shape[-2:]), **self.dd
        )
        logger.debug(
            "Polarizability derivative (numerical): Starting build (%s).",
            deriv.shape,
        )

        count = 1
        nsteps = 3 * self.numbers.shape[-1]
        for i in range(self.numbers.shape[-1]):
            for j in range(3):
                with OutputHandler.with_verbosity(0):
                    positions[..., i, j] += step_size
                    r = self.polarizability_numerical(positions, chrg, spin)

                    positions[..., i, j] -= 2 * step_size
                    l = self.polarizability_numerical(positions, chrg, spin)

                    positions[..., i, j] += step_size
                    deriv[..., :, :, i, j] = 0.5 * (r - l) / step_size

                logger.debug(
                    "Polarizability numerical derivative: Step %s/%s",
                    count,
                    nsteps,
                )
                count += 1

                gc.collect()
            gc.collect()

        logger.debug("Polarizability numerical derivative: All finished.")

        return deriv

    @cdec.numerical
    @cdec.requires_efield
    @cdec.cache
    def hyperpolarizability_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> Tensor:
        r"""
        Numerically calculate the hyper polarizability tensor :math:`\beta`.

        .. math::

            \beta = \dfrac{\partial \alpha}{\partial F}
            = \dfrac{\partial^2 \mu}{\partial F^2}
            = \dfrac{\partial^3 E}{\partial^2 3}

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int, optional
            Total charge. Defaults to 0.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        Tensor
            Hyper polarizability tensor of shape ``(..., 3, 3, 3)``.
        """
        # pylint: disable=import-outside-toplevel
        import gc

        # retrieve the efield interaction and the field and detach for gradient
        ef = self.interactions.get_interaction(efield.LABEL_EFIELD)
        _field = ef.field.clone()
        field = ef.field.detach().clone()
        self.interactions.update_efield(field=field)

        # (..., 3, 3, 3)
        deriv = torch.zeros(*(*self.numbers.shape[:-1], 3, 3, 3), **self.dd)
        logger.debug(
            "Hyper Polarizability (numerical): Starting build (%s)", deriv.shape
        )

        count = 1
        for i in range(3):
            with OutputHandler.with_verbosity(0):
                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                gr = self.polarizability_numerical(positions, chrg, spin)

                field[..., i] -= 2 * step_size
                self.interactions.update_efield(field=field)
                gl = self.polarizability_numerical(positions, chrg, spin)

                field[..., i] += step_size
                self.interactions.update_efield(field=field)
                deriv[..., :, :, i] = 0.5 * (gr - gl) / step_size

            logger.debug("Hyper Polarizability (numerical): step %s/3", count)
            count += 1

            gc.collect()

        # explicitly update field (to restore original field with possible grad)
        self.interactions.reset_efield()
        self.interactions.update_efield(field=_field)

        logger.debug("Hyper Polarizability (numerical): All finished.")

        return deriv

    # SPECTRA

    @cdec.numerical
    @cdec.cache
    def ir_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> vib.IRResult:
        """
        Numerically calculate the frequencies and intensities of IR spectra.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        vib.IRResult
            Result container with frequencies (shape: ``(..., nfreqs)``) and
            intensities (shape: ``(..., nfreqs)``) of IR spectra.
        """
        OutputHandler.write_stdout("\nIR Spectrum")
        OutputHandler.write_stdout("-----------")
        logger.debug("IR spectrum (numerical): Start.")

        # run vibrational analysis first
        freqs, modes = self.vibration_numerical(
            positions, chrg, spin, step_size=step_size
        )

        # calculate nuclear dipole derivative dmu/dR: (..., 3, nat, 3)
        dmu_dr = self.dipole_deriv_numerical(positions, chrg, spin, step_size=step_size)

        intensities = vib.ir_ints(dmu_dr, modes)

        logger.debug("IR spectrum (numerical): All finished.")

        return vib.IRResult(freqs, intensities)

    @cdec.numerical
    @cdec.cache
    def raman_numerical(
        self,
        positions: Tensor,
        chrg: Tensor | float | int = defaults.CHRG,
        spin: Tensor | float | int | None = defaults.SPIN,
        step_size: int | float = defaults.STEP_SIZE,
    ) -> vib.RamanResult:
        """
        Numerically calculate the frequencies, static intensities and
        depolarization ratio of Raman spectra.
        Formula taken from `here <https://doi.org/10.1080/00268970701516412>`__.

        Parameters
        ----------
        positions : Tensor
            Cartesian coordinates of all atoms (shape: ``(..., nat, 3)``).
        chrg : Tensor | float | int | str | None
            Total charge. Defaults to ``None``.
        spin : Tensor | float | int, optional
            Number of unpaired electrons. Defaults to ``None``.
        step_size : float | int, optional
            Step size for the numerical derivative.

        Returns
        -------
        vib.RamanResult
            Result container with frequencies (shape: ``(..., nfreqs)``),
            intensities (shape: ``(..., nfreqs)``) and the depolarization ratio
            (shape: ``(..., nfreqs)``) of Raman spectra.
        """
        OutputHandler.write_stdout("\nRaman Spectrum")
        OutputHandler.write_stdout("--------------")
        logger.debug("Raman spectrum (numerical): All finished.")

        vib_res = self.vibration_numerical(positions, chrg, spin, step_size=step_size)

        # d(3, 3) / d(nat, 3) -> (3, 3, nat, 3) -> (3, 3, nat*3)
        da_dr = self.pol_deriv_numerical(positions, chrg, spin, step_size=step_size)

        intensities, depol = vib.raman_ints_depol(da_dr, vib_res.modes)

        logger.debug("Raman spectrum: All finished.")

        return vib.RamanResult(vib_res.freqs, intensities, depol)

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
        super().calculate(properties, positions, chrg, spin, **kwargs)

        if "forces" in properties:
            self.forces_numerical(positions, chrg, spin, **kwargs)

        if "hessian" in properties:
            self.hessian_numerical(positions, chrg, spin, **kwargs)

        if {"vibration", "frequencies", "normal_modes"} & set(properties):
            self.vibration_numerical(positions, chrg, spin, **kwargs)

        if "dipole" in properties:
            self.dipole_numerical(positions, chrg, spin, **kwargs)

        if {"dipole_derivatives", "dipole_deriv"} & set(properties):
            self.dipole_deriv_numerical(positions, chrg, spin, **kwargs)

        if "polarizability" in properties:
            self.polarizability_numerical(positions, chrg, spin, **kwargs)

        if {"polarizability_derivatives", "pol_deriv"} & set(properties):
            self.pol_deriv_numerical(positions, chrg, spin, **kwargs)

        if "hyperpolarizability" in properties:
            self.hyperpolarizability_numerical(positions, chrg, spin, **kwargs)

        if {"ir", "ir_intensities"} in set(properties):
            self.ir_numerical(positions, chrg, spin, **kwargs)

        if {"raman", "raman_intensities", "raman_depol"} & set(properties):
            self.raman_numerical(positions, chrg, spin, **kwargs)
