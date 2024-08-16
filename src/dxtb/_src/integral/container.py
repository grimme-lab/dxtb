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
Integral container
==================

A class that acts as a container for integrals.
"""

from __future__ import annotations

import logging

import torch

from dxtb import labels
from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any, Tensor
from dxtb._src.xtb.base import BaseHamiltonian

from .base import IntegralContainer
from .driver import DriverManager
from .types import DipoleIntegral, OverlapIntegral, QuadrupoleIntegral

__all__ = ["Integrals", "IntegralMatrices"]

logger = logging.getLogger(__name__)


class Integrals(IntegralContainer):
    """
    Integral container.
    """

    __slots__ = [
        "_hcore",
        "_overlap",
        "_dipole",
        "_quadrupole",
        "_driver",
    ]

    def __init__(
        self,
        driver_manager: DriverManager,
        *,
        intlevel: int = defaults.INTLEVEL,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _hcore: BaseHamiltonian | None = None,
        _overlap: OverlapIntegral | None = None,
        _dipole: DipoleIntegral | None = None,
        _quadrupole: QuadrupoleIntegral | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.driver_manager = driver_manager

        self._hcore = _hcore
        self._overlap = _overlap
        self._dipole = _dipole
        self._quadrupole = _quadrupole
        self._intlevel = intlevel

    # Core Hamiltonian

    @property
    def hcore(self) -> BaseHamiltonian | None:
        return self._hcore

    @hcore.setter
    def hcore(self, hcore: BaseHamiltonian) -> None:
        self._hcore = hcore
        self.checks()

    # TODO: Allow Hamiltonian build without overlap
    def build_hcore(
        self, positions: Tensor, overlap: Tensor | None = None, **kwargs
    ) -> Tensor:
        logger.debug("Core Hamiltonian: Start building matrix.")

        if self.hcore is None:
            raise RuntimeError("Core Hamiltonian integral not initialized.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        # overlap integral required
        if overlap is None:
            overlap = self.build_overlap(positions, **kwargs)

        cn = kwargs.pop("cn", None)
        if cn is None:
            # pylint: disable=import-outside-toplevel
            from ..ncoord import cn_d3

            cn = cn_d3(self.hcore.numbers, positions)

        hcore = self.hcore.build(positions, self.overlap.matrix, cn=cn)
        logger.debug("Core Hamiltonian: All finished.")
        return hcore

    # overlap

    @property
    def overlap(self) -> OverlapIntegral | None:
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: OverlapIntegral) -> None:
        self._overlap = overlap
        self.checks()

    def build_overlap(self, positions: Tensor, **kwargs: Any) -> Tensor:
        # in case CPU is forced for libcint, move positions to CPU
        if self.driver_manager.force_cpu_for_libcint is True:
            positions = positions.to(device=torch.device("cpu"))

        self.driver_manager.setup_driver(positions, **kwargs)
        logger.debug("Overlap integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("No overlap integral class provided.")

        self.overlap.build(self.driver_manager.driver)
        assert self.overlap.matrix is not None

        # move integral to the correct device...
        if self.driver_manager.force_cpu_for_libcint is True:
            # ... but only if no other multipole integrals are required
            if self._intlevel <= labels.INTLEVEL_HCORE:
                self.overlap.matrix = self.overlap.matrix.to(device=self.device)

                # FIXME: The matrix has to be moved explicitly, because when
                # singlepoint is called a second time, the integral is already
                # on the correct device (from the to of the first call) and the
                # matrix is not moved because the to method exits immediately.
                # This is a workaround and can possibly be fixed when the
                # matrices are no longer stored (should only return in sp)

        logger.debug("Overlap integral: All finished.")

        return self.overlap.matrix

    def grad_overlap(self, positions: Tensor, **kwargs) -> Tensor:
        # in case CPU is forced for libcint, move positions to CPU
        if self.driver_manager.force_cpu_for_libcint is True:
            positions = positions.to(device=torch.device("cpu"))

        self.driver_manager.setup_driver(positions, **kwargs)

        if self.overlap is None:
            raise RuntimeError("No overlap integral provided.")

        logger.debug("Overlap gradient: Start.")
        grad = self.overlap.get_gradient(self.driver_manager.driver, **kwargs)
        logger.debug("Overlap gradient: All finished.")

        return grad.to(self.device)

    # dipole

    @property
    def dipole(self) -> DipoleIntegral | None:
        """
        Dipole integral of shape (3, nao, nao).

        Returns
        -------
        Tensor | None
            Dipole integral if set, else ``None``.
        """
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: DipoleIntegral) -> None:
        self._dipole = dipole
        self.checks()

    def build_dipole(self, positions: Tensor, shift: bool = True, **kwargs: Any):
        # in case CPU is forced for libcint, move positions to CPU
        if self.driver_manager.force_cpu_for_libcint:
            positions = positions.to(device=torch.device("cpu"))

        self.driver_manager.setup_driver(positions, **kwargs)
        logger.debug("Dipole integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral class not initialized.")

        if self.dipole is None:
            raise RuntimeError("Dipole integral class not initialized.")

        # build (with overlap norm)
        self.dipole.build(self.driver_manager.driver)
        self.dipole.normalize(self.overlap.norm)
        logger.debug("Dipole integral: Finished building matrix.")

        # shift to rj (requires overlap integral)
        if shift is True:
            logger.debug("Dipole integral: Start shifting operator (r0->rj).")
            self.dipole.shift_r0_rj(
                self.overlap.matrix,
                self.driver_manager.driver.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )
            logger.debug("Dipole integral: Finished shifting operator.")

        # move integral to the correct device, but only if no other multipole
        # integrals are required
        if (
            self.driver_manager.force_cpu_for_libcint
            and self._intlevel <= labels.INTLEVEL_DIPOLE
        ):
            self.dipole.matrix = self.dipole.matrix.to(device=self.device)
            self.overlap.matrix = self.overlap.matrix.to(device=self.device)

        logger.debug("Dipole integral: All finished.")
        return self.dipole.matrix

    # quadrupole

    @property
    def quadrupole(self) -> QuadrupoleIntegral | None:
        """
        Quadrupole integral of shape (6/9, nao, nao).

        Returns
        -------
        Tensor | None
            Quadrupole integral if set, else ``None``.
        """
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, quadrupole: QuadrupoleIntegral) -> None:
        self._quadrupole = quadrupole
        self.checks()

    def build_quadrupole(
        self,
        positions: Tensor,
        shift: bool = True,
        traceless: bool = True,
        **kwargs: Any,
    ):
        # in case CPU is forced for libcint, move positions to CPU
        if self.driver_manager.force_cpu_for_libcint:
            positions = positions.to(device=torch.device("cpu"))

        # check all instantiations
        self.driver_manager.setup_driver(positions, **kwargs)
        logger.debug("Quad integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        if self.quadrupole is None:
            raise RuntimeError("Quadrupole integral not initialized.")

        # build
        self.quadrupole.build(self.driver_manager.driver)
        self.quadrupole.normalize(self.overlap.norm)
        logger.debug("Quad integral: Finished building matrix.")

        # make traceless before shifting
        if traceless is True:
            logger.debug("Quad integral: Start creating traceless rep.")
            self.quadrupole.traceless()
            logger.debug("Quad integral: Finished creating traceless rep.")

        # shift to rj (requires overlap and dipole integral)
        if shift is True:
            logger.debug("Quad integral: Start shifting operator (r0r0->rjrj).")
            if traceless is not True:
                raise RuntimeError(
                    "Quadrupole moment must be tracelesss for shifting. "
                    "Run `quadrupole.traceless()` before shifting."
                )

            if self.dipole is None:
                self.build_dipole(positions, **kwargs)
            assert self.dipole is not None

            self.quadrupole.shift_r0r0_rjrj(
                self.dipole.matrix,
                self.overlap.matrix,
                self.driver_manager.driver.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )
            logger.debug("Quad integral: Finished shifting operator.")

        # Finally, we move the integral to the correct device, but only if
        # no other multipole integrals are required.
        if (
            self.driver_manager.force_cpu_for_libcint
            and self._intlevel <= labels.INTLEVEL_QUADRUPOLE
        ):
            self.overlap.matrix = self.overlap.matrix.to(self.device)
            self.quadrupole.matrix = self.quadrupole.matrix.to(self.device)

            if self.dipole is not None:
                self.dipole.matrix = self.dipole.matrix.to(self.device)

        logger.debug("Quad integral: All finished.")
        return self.quadrupole.matrix

    # checks

    def checks(self) -> None:
        if self.run_checks is False:
            return

        for name in ["hcore", "overlap", "dipole", "quadrupole"]:
            cls: (
                BaseHamiltonian
                | OverlapIntegral
                | DipoleIntegral
                | QuadrupoleIntegral
                | None
            ) = getattr(self, f"_{name}")

            if cls is None:
                continue

            if cls.dtype != self.dtype:
                raise RuntimeError(
                    f"Data type of '{cls.label}' integral ({cls.dtype}) and "
                    f"integral container {self.dtype} do not match."
                )
            if cls.device != self.device:
                raise RuntimeError(
                    f"Device of '{cls.label}' integral ({cls.device}) and "
                    f"integral container {self.device} do not match."
                )

            if name != "hcore":
                assert not isinstance(cls, BaseHamiltonian)

                print(cls.family)
                family_integral = cls.family
                family_driver = self.driver_manager.driver.family
                driver_label = self.driver_manager.driver
                if family_integral != family_driver:
                    raise RuntimeError(
                        f"The '{cls.label}' integral implementation "
                        f"requests the '{family_integral}' family, but "
                        f"the integral driver '{driver_label}' is "
                        "configured.\n"
                        "If you want to request the 'pytorch' implementations, "
                        "specify the driver name in the constructors of both "
                        "the integral container and the actual integral class."
                    )

    def reset_all(self) -> None:
        self.driver_manager.invalidate_driver()
        # TODO: Do we need to reset the specific integrals?

    # pretty print

    def __str__(self) -> str:
        attributes = ["hcore", "overlap", "dipole", "quadrupole"]
        details = []

        for attr in attributes:
            i = getattr(self, "_" + attr)
            info = str(i) if i is not None else "None"
            details.append(f"\n  {attr}={info}")

        return f"Integrals({', '.join(details)}\n)"

    def __repr__(self) -> str:
        return str(self)


class IntegralMatrices(IntegralContainer):
    """
    Storage container for the integral matrices.
    """

    __slots__ = ["_hcore", "_overlap", "_dipole", "_quadrupole"]

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        _hcore: Tensor | None = None,
        _overlap: Tensor | None = None,
        _dipole: Tensor | None = None,
        _quadrupole: Tensor | None = None,
        _run_checks: bool = True,
    ):
        super().__init__(device, dtype, _run_checks)

        self._hcore = _hcore
        self._overlap = _overlap
        self._dipole = _dipole
        self._quadrupole = _quadrupole

    # Core Hamiltonian

    @property
    def hcore(self) -> Tensor:
        if self._hcore is None:
            raise RuntimeError("Core Hamiltonian matrix not set.")
        return self._hcore

    @hcore.setter
    def hcore(self, mat: Tensor) -> None:
        self._hcore = mat
        self.checks()
        self.device_check()

    # overlap

    @property
    def overlap(self) -> Tensor:
        if self._overlap is None:
            raise RuntimeError("Overlap matrix not set.")
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: Tensor) -> None:
        self._overlap = overlap
        self.checks()
        self.device_check()

    # dipole

    @property
    def dipole(self) -> Tensor | None:
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: Tensor) -> None:
        self._dipole = dipole
        self.checks()
        self.device_check()

    # quadrupole

    @property
    def quadrupole(self) -> Tensor | None:
        """
        Quadrupole integral of shape (6/9, nao, nao).

        Returns
        -------
        Tensor | None
            Quadrupole integral if set, else ``None``.
        """
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, mat: Tensor) -> None:
        self._quadrupole = mat
        self.checks()
        self.device_check()

    # checks

    def device_check(self) -> None:
        """
        Check if all tensors are on the same device after calling a setter.
        If not, set the device to ``"invalid"``. Otherwise, the :meth:`to`
        method is not triggered properly.
        """

        for name in ["hcore", "overlap", "dipole", "quadrupole"]:
            tensor: Tensor = getattr(self, "_" + name)
            if tensor is None:
                continue

            if tensor.device != self.device:
                self.override_device("invalid")  # type: ignore
                break

    def checks(self) -> None:
        """
        Checks the shapes of the tensors.

        Expected shapes:

        - hcore and overlap: (batch_size, nao, nao) or (nao, nao)
        - dipole: (batch_size, 3, nao, nao) or (3, nao, nao)
        - quad: (batch_size, 9, nao, nao) or (9, nao, nao)

        Raises
        ------
        ValueError:
            If any of the tensors have incorrect shapes or inconsistent batch
            sizes.
        """
        if self.run_checks is False:
            return

        nao = None
        batch_size = None

        for name in ["hcore", "overlap", "dipole", "quadrupole"]:
            tensor: Tensor = getattr(self, "_" + name)
            if tensor is None:
                continue

            if name in ["hcore", "overlap"]:
                if len(tensor.shape) not in [2, 3]:
                    raise ValueError(
                        f"Tensor '{name}' must have 2 or 3 dimensions. "
                        f"Got {len(tensor.shape)}."
                    )
                if len(tensor.shape) == 3:
                    if batch_size is not None and tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"Tensor '{name}' has a different batch size. "
                            f"Expected {batch_size}, got {tensor.shape[0]}."
                        )
                    batch_size = tensor.shape[0]
                nao = tensor.shape[-1]
            elif name in ["dipole", "quadrupole"]:
                if len(tensor.shape) not in [3, 4]:
                    raise ValueError(
                        f"Tensor '{name}' must have 3 or 4 dimensions. "
                        f"Got {len(tensor.shape)}."
                    )
                if len(tensor.shape) == 4:
                    if batch_size is not None and tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"Tensor '{name}' has a different batch size. "
                            f"Expected {batch_size}, got {tensor.shape[0]}."
                        )
                    batch_size = tensor.shape[0]
                nao = tensor.shape[-2]

            if tensor.shape[-2:] != (nao, nao):
                raise ValueError(
                    f"Tensor '{name}' last two dimensions should be "
                    f"(nao, nao). Got {tensor.shape[-2:]}."
                )
            if name == "dipole" and tensor.shape[-3] != defaults.DP_SHAPE:
                raise ValueError(
                    f"Tensor '{name}' third to last dimension should be "
                    f"{defaults.DP_SHAPE}. Got {tensor.shape[-3]}."
                )
            if "quad" in name and tensor.shape[-3] != defaults.QP_SHAPE:
                raise ValueError(
                    f"Tensor '{name}' third to last dimension should be "
                    f"{defaults.QP_SHAPE}. Got {tensor.shape[-3]}."
                )

    def __str__(self) -> str:
        attributes = ["hcore", "overlap", "dipole", "quadrupole"]
        details = []

        for attr in attributes:
            tensor = getattr(self, "_" + attr)
            info = str(tensor.shape) if tensor is not None else "None"
            details.append(f"\n  {attr}={info}")

        return f"Integrals({', '.join(details)}\n)"

    def __repr__(self) -> str:
        return str(self)
