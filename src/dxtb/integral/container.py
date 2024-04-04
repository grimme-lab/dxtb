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

from dxtb.basis import IndexHelper
from dxtb.typing import Any, Tensor

from ..constants import defaults, labels
from ..param import Param
from .base import IntDriver, IntegralContainer
from .dipole import Dipole
from .driver import IntDriverLibcint, IntDriverPytorch, IntDriverPytorchNoAnalytical
from .h0 import Hamiltonian
from .overlap import Overlap
from .quadrupole import Quadrupole

__all__ = ["Integrals", "IntegralMatrices"]

logger = logging.getLogger(__name__)


class Integrals(IntegralContainer):
    """
    Integral container.
    """

    __slots__ = [
        "numbers",
        "par",
        "ihelp",
        "_hcore",
        "_overlap",
        "_dipole",
        "_quadrupole",
        "_matrices",
        "_run_checks",
        "_driver",
    ]

    def __init__(
        self,
        numbers: Tensor,
        par: Param,
        ihelp: IndexHelper,
        *,
        hcore: Hamiltonian | None = None,
        overlap: Overlap | None = None,
        dipole: Dipole | None = None,
        quadrupole: Quadrupole | None = None,
        driver: int = labels.INTDRIVER_LIBCINT,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        self.numbers = numbers
        self.par = par
        self.ihelp = ihelp
        self._hcore = hcore
        self._overlap = overlap
        self._dipole = dipole
        self._quadrupole = quadrupole
        self._matrices = IntegralMatrices(device=device, dtype=dtype)

        # Determine which driver class to instantiate
        if driver == labels.INTDRIVER_LIBCINT:
            self._driver = IntDriverLibcint(
                numbers, par, ihelp, device=device, dtype=dtype
            )
        elif driver == labels.INTDRIVER_ANALYTICAL:
            self._driver = IntDriverPytorch(
                numbers, par, ihelp, device=device, dtype=dtype
            )
        elif driver == labels.INTDRIVER_AUTOGRAD:
            self._driver = IntDriverPytorchNoAnalytical(
                numbers, par, ihelp, device=device, dtype=dtype
            )
        else:
            raise ValueError(f"Unknown integral driver '{driver}'.")

    @property
    def matrices(self) -> IntegralMatrices:
        return self._matrices

    @matrices.setter
    def matrices(self, matrices: IntegralMatrices) -> None:
        self._matrices = matrices

    # Integral driver

    @property
    def driver(self) -> IntDriver:
        if self._driver is None:
            raise ValueError("No integral driver provided.")
        return self._driver

    @driver.setter
    def driver(self, driver: IntDriver) -> None:
        self._driver = driver

    def setup_driver(self, positions: Tensor, **kwargs: Any) -> None:
        logger.debug("Integral Driver: Start setup.")
        if self.driver.is_latest(positions) is True:
            logger.debug("Integral Driver: Skip setup. Already done.")
            return

        self.driver.setup(positions, **kwargs)
        logger.debug("Integral Driver: Finished setup.")

    def invalidate_driver(self) -> None:
        """Invalidate the integral driver to require new setup."""
        self.driver.invalidate()

    # Core Hamiltonian

    @property
    def hcore(self) -> Hamiltonian | None:
        return self._hcore

    @hcore.setter
    def hcore(self, hcore: Hamiltonian) -> None:
        self._hcore = hcore
        self.checks()

    # TODO: Allow Hamiltonian build without overlap
    def build_hcore(self, positions: Tensor, **kwargs) -> Tensor:
        logger.debug("Core Hamiltonian: Start building matrix.")

        if self.hcore is None:
            raise RuntimeError("Core Hamiltonian integral not initialized.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        # overlap integral required
        ovlp = self.overlap.integral
        if ovlp.matrix is None:
            self.build_overlap(positions, **kwargs)

        cn = kwargs.pop("cn", None)
        if cn is None:
            # pylint: disable=import-outside-toplevel
            from ..ncoord import cn_d3

            cn = cn_d3(self.numbers, positions)

        hcore = self.hcore.integral.build(positions, ovlp.matrix, cn=cn)
        self._matrices.hcore = hcore
        logger.debug("Core Hamiltonian: All finished.")
        return hcore

    # overlap

    @property
    def overlap(self) -> Overlap | None:
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: Overlap) -> None:
        self._overlap = overlap
        self.checks()

    def build_overlap(self, positions: Tensor, **kwargs: Any) -> Tensor:
        self.setup_driver(positions, **kwargs)
        logger.debug("Overlap integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("No overlap integral provided.")

        overlap = self.overlap.build(self.driver)
        self._matrices.overlap = overlap
        logger.debug("Overlap integral: All finished.")
        return overlap

    def grad_overlap(self, positions: Tensor, **kwargs) -> Tensor:
        self.setup_driver(positions, **kwargs)

        if self.overlap is None:
            raise RuntimeError("No overlap integral provided.")
        return self.overlap.get_gradient(self.driver, **kwargs)

    # dipole

    @property
    def dipole(self) -> Dipole | None:
        """
        Dipole integral of shape (3, nao, nao).

        Returns
        -------
        Tensor | None
            Dipole integral if set, else `None`.
        """
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: Dipole) -> None:
        self._dipole = dipole
        self.checks()

    def build_dipole(self, positions: Tensor, shift: bool = True, **kwargs: Any):
        self.setup_driver(positions, **kwargs)
        logger.debug("Dipole integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        if self.dipole is None:
            raise RuntimeError("Dipole integral not initialized.")

        # build (with overlap norm)
        self.dipole.integral.norm = self._norm(positions)
        self.dipole.build(self.driver)
        logger.debug("Dipole integral: Finished building matrix.")

        # shift to rj (requires overlap integral)
        if shift is True:
            logger.debug("Dipole integral: Start shifting operator (r0->rj).")
            self.dipole.integral.shift_r0_rj(
                self.overlap.integral.matrix,
                self.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )
            logger.debug("Dipole integral: Finished shifting operator.")

        self._matrices.dipole = self.dipole.integral.matrix
        logger.debug("Dipole integral: All finished.")
        return self.dipole.integral.matrix

    # quadrupole

    @property
    def quadrupole(self) -> Quadrupole | None:
        """
        Quadrupole integral of shape (6/9, nao, nao).

        Returns
        -------
        Tensor | None
            Quadrupole integral if set, else `None`.
        """
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, quadrupole: Quadrupole) -> None:
        self._quadrupole = quadrupole
        self.checks()

    def build_quadrupole(
        self,
        positions: Tensor,
        shift: bool = True,
        traceless: bool = True,
        **kwargs: Any,
    ):
        # check all instantiations
        self.setup_driver(positions, **kwargs)
        logger.debug("Quad integral: Start building matrix.")

        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        if self.quadrupole is None:
            raise RuntimeError("Quadrupole integral not initialized.")

        # build
        self.quadrupole.integral.norm = self._norm(positions, **kwargs)
        self.quadrupole.build(self.driver)
        logger.debug("Quad integral: Finished building matrix.")

        # make traceless before shifting
        if traceless is True:
            logger.debug("Quad integral: Start creating traceless rep.")
            self.quadrupole.integral.traceless()
            logger.debug("Quad integral: Finished creating traceless rep.")

        # shift to rj (requires overlap and dipole integral)
        if shift is True:
            logger.debug("Quad integral: Start shifting operator (r0r0->rjrj).")
            if traceless is not True:
                raise RuntimeError("Quadrupole moment must be tracelesss for shifting.")

            if self.dipole is None:
                self.build_dipole(positions, **kwargs)
            assert self.dipole is not None

            self.quadrupole.integral.shift_r0r0_rjrj(
                self.dipole.integral.matrix,
                self.overlap.integral.matrix,
                self.ihelp.spread_atom_to_orbital(
                    positions,
                    dim=-2,
                    extra=True,
                ),
            )
            logger.debug("Quad integral: Finished shifting operator.")

        self._matrices.quadrupole = self.quadrupole.integral.matrix
        logger.debug("Quad integral: All finished.")
        return self.quadrupole.integral.matrix

    # helper

    def _norm(self, positions: Tensor, **kwargs: Any) -> Tensor:
        if self.overlap is None:
            raise RuntimeError("Overlap integral not initialized.")

        # shortcut for overlap integral
        ovlp = self.overlap.integral

        # overlap integral required for norm and shifting
        if ovlp.matrix is None or ovlp.norm is None:
            self.build_overlap(positions, **kwargs)

        return ovlp.norm

    # checks

    def checks(self) -> None:
        if self.run_checks is False:
            return

        for name in ["hcore", "overlap", "dipole", "quadrupole"]:
            cls = getattr(self, "_" + name)
            if cls is None:
                continue

            cls: Hamiltonian | Overlap | Dipole | Quadrupole

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
                family_integral = cls.integral.family  # type: ignore
                family_driver = self.driver.family
                if family_integral != family_driver:
                    raise RuntimeError(
                        f"The '{cls.integral.label}' integral implementation "
                        f"requests the '{family_integral}' family, but "
                        f"the integral driver '{self.driver.label}' is "
                        f"configured with the '{family_driver}' family.\n"
                        "If you want to request the 'pytorch' implementations, "
                        "specify the driver name in the constructors of both "
                        "the integral container and the actual integral class."
                    )

    def reset_all(self) -> None:
        self.invalidate_driver()
        # TODO: Do we need to reset the specific integrals and the
        # IntegralMatrices?

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

    __slots__ = ["_hcore", "_overlap", "_dipole", "_quadrupole", "_run_checks"]

    def __init__(
        self,
        hcore: Tensor | None = None,
        overlap: Tensor | None = None,
        dipole: Tensor | None = None,
        quadrupole: Tensor | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(device, dtype)

        self._hcore = hcore
        self._overlap = overlap
        self._dipole = dipole
        self._quadrupole = quadrupole

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

    # dipole

    @property
    def dipole(self) -> Tensor | None:
        return self._dipole

    @dipole.setter
    def dipole(self, dipole: Tensor) -> None:
        self._dipole = dipole
        self.checks()

    # quadrupole

    @property
    def quadrupole(self) -> Tensor | None:
        """
        Quadrupole integral of shape (6/9, nao, nao).

        Returns
        -------
        Tensor | None
            Quadrupole integral if set, else `None`.
        """
        return self._quadrupole

    @quadrupole.setter
    def quadrupole(self, mat: Tensor) -> None:
        self._quadrupole = mat
        self.checks()

    # checks

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
