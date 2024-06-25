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
SCF configuration.
"""

from __future__ import annotations

import torch

from dxtb import OutputHandler
from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any, get_default_device, get_default_dtype

__all__ = ["ConfigSCF", "ConfigFermi"]


class ConfigSCF:
    """
    Configuration for the SCF.

    All configuration options are represented as integers. String options are
    converted to integers in the constructor.

    The settings for Fermi smearing are stored separately in the
    :class:`ConfigFermi` class, which can be accessed via the :attr:`fermi`
    attribute.
    """

    strict: bool = False
    """Strict mode for SCF configuration. Always throws errors if ``True``."""

    guess: int
    """Initial guess for the SCF."""

    maxiter: int
    """Maximum number of SCF iterations."""

    mixer: int
    """Mixing scheme for SCF iterations."""

    scf_mode: int
    """SCF convergence approach (denoted by backward strategy)."""

    scp_mode: int
    """SCF convergence target (self-consistent property)."""

    x_atol: float
    """Absolute tolerance for argument (x) in SCF solver."""

    f_atol: float
    """Absolute tolerance for function value (f(x)) the SCF solver."""

    force_convergence: bool
    """Force convergence of the SCF iterations."""

    batch_mode: int
    """Batch mode for the SCF iterations."""

    # Fermi

    fermi_etemp: float
    """Electronic temperature for Fermi smearing."""

    fermi_maxiter: int
    """Maximum number of iterations for Fermi smearing."""

    fermi_thresh: dict
    """Threshold for Fermi iterations."""

    fermi_partition: int
    """Partitioning scheme for electronic free energy."""

    # PyTorch

    device: torch.device
    """Device for calculations."""

    dtype: torch.dtype
    """Data type for calculations."""

    def __init__(
        self,
        *,
        strict: bool = False,
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        mixer: str = defaults.MIXER,
        damp: float = defaults.DAMP,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        x_atol: float = defaults.X_ATOL,
        f_atol: float = defaults.F_ATOL,
        force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE,
        batch_mode: int = defaults.BATCH_MODE,
        # Fermi
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: dict = defaults.FERMI_THRESH,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
        # PyTorch
        device: torch.device = get_default_device(),
        dtype: torch.dtype = get_default_dtype(),
    ) -> None:
        self.strict = strict

        if isinstance(guess, str):
            if guess.casefold() in labels.GUESS_EEQ_STRS:
                self.guess = labels.GUESS_EEQ
            elif guess.casefold() in labels.GUESS_SAD_STRS:
                self.guess = labels.GUESS_SAD
            else:
                guess_labels = labels.GUESS_EEQ_STRS + labels.GUESS_SAD_STRS
                raise ValueError(
                    f"Unknown guess method '{guess}'. "
                    f"Use one of '{', '.join(guess_labels)}'."
                )
        elif isinstance(guess, int):
            if guess not in (labels.GUESS_EEQ, labels.GUESS_SAD):
                guess_labels = labels.GUESS_EEQ_STRS + labels.GUESS_SAD_STRS
                raise ValueError(
                    f"Unknown guess method '{guess}'. "
                    f"Use one of '{', '.join(guess_labels)}'."
                )

            self.guess = guess
        else:
            raise TypeError(
                "The guess must be of type 'int' or 'str', but "
                f"'{type(guess)}' was given."
            )

        if isinstance(scf_mode, str):
            if scf_mode.casefold() in labels.SCF_MODE_IMPLICIT_STRS:
                self.scf_mode = labels.SCF_MODE_IMPLICIT
            elif scf_mode.casefold() in labels.SCF_MODE_IMPLICIT_NON_PURE_STRS:
                self.scf_mode = labels.SCF_MODE_IMPLICIT_NON_PURE
            elif scf_mode.casefold() in labels.SCF_MODE_FULL_STRS:
                self.scf_mode = labels.SCF_MODE_FULL
            elif scf_mode.casefold() in labels.SCF_MODE_EXPERIMENTAL_STRS:
                self.scf_mode = labels.SCF_MODE_EXPERIMENTAL
            else:
                scf_mode_labels = (
                    labels.SCF_MODE_IMPLICIT_STRS
                    + labels.SCF_MODE_IMPLICIT_NON_PURE_STRS
                    + labels.SCF_MODE_FULL_STRS
                    + labels.SCF_MODE_EXPERIMENTAL_STRS
                )
                raise ValueError(
                    f"Unknown SCF mode '{scf_mode}'. "
                    f"Use one of '{', '.join(scf_mode_labels)}'."
                )
        elif isinstance(scf_mode, int):
            if scf_mode not in (
                labels.SCF_MODE_IMPLICIT,
                labels.SCF_MODE_IMPLICIT_NON_PURE,
                labels.SCF_MODE_FULL,
                labels.SCF_MODE_EXPERIMENTAL,
            ):
                scf_mode_labels = (
                    labels.SCF_MODE_IMPLICIT_STRS
                    + labels.SCF_MODE_IMPLICIT_NON_PURE_STRS
                    + labels.SCF_MODE_FULL_STRS
                    + labels.SCF_MODE_EXPERIMENTAL_STRS
                )
                raise ValueError(
                    f"Unknown SCF mode '{scf_mode}'. "
                    f"Use one of '{', '.join(scf_mode_labels)}'."
                )

            self.scf_mode = scf_mode

        else:
            raise TypeError(
                "The scf_mode must be of type 'int' or 'str', but "
                f"'{type(scf_mode)}' was given."
            )

        if isinstance(scp_mode, str):
            if scp_mode.casefold() in labels.SCP_MODE_CHARGE_STRS:
                self.scp_mode = labels.SCP_MODE_CHARGE
            elif scp_mode.casefold() in labels.SCP_MODE_POTENTIAL_STRS:
                self.scp_mode = labels.SCP_MODE_POTENTIAL
            elif scp_mode.casefold() in labels.SCP_MODE_FOCK_STRS:
                self.scp_mode = labels.SCP_MODE_FOCK
            else:
                scp_mode_labels = (
                    labels.SCP_MODE_CHARGE_STRS
                    + labels.SCP_MODE_POTENTIAL_STRS
                    + labels.SCP_MODE_FOCK_STRS
                )
                raise ValueError(
                    f"Unknown convergence target (SCP mode) '{scp_mode}'. "
                    f"Use one of '{', '.join(scp_mode_labels)}'."
                )
        elif isinstance(scp_mode, int):
            if scp_mode not in (
                labels.SCP_MODE_CHARGE,
                labels.SCP_MODE_POTENTIAL,
                labels.SCP_MODE_FOCK,
            ):
                scp_mode_labels = (
                    labels.SCP_MODE_CHARGE_STRS
                    + labels.SCP_MODE_POTENTIAL_STRS
                    + labels.SCP_MODE_FOCK_STRS
                )
                raise ValueError(
                    f"Unknown convergence target (SCP mode) '{scp_mode}'. "
                    f"Use one of '{', '.join(scp_mode_labels)}'."
                )

            self.scp_mode = scp_mode
        else:
            raise TypeError(
                "The scp_mode must be of type 'int' or 'str', but "
                f"'{type(scp_mode)}' was given."
            )

        if isinstance(mixer, str):
            if mixer.casefold() in labels.MIXER_LINEAR_STRS:
                self.mixer = labels.MIXER_LINEAR
            elif mixer.casefold() in labels.MIXER_ANDERSON_STRS:
                self.mixer = labels.MIXER_ANDERSON
            elif mixer.casefold() in labels.MIXER_BROYDEN_STRS:
                self.mixer = labels.MIXER_BROYDEN
            else:
                mixer_labels = (
                    labels.MIXER_LINEAR_STRS
                    + labels.MIXER_ANDERSON_STRS
                    + labels.MIXER_BROYDEN_STRS
                )
                raise ValueError(
                    f"Unknown mixer '{mixer}'. Choose from "
                    f"'{', '.join(mixer_labels)}'."
                )
        elif isinstance(mixer, int):
            if mixer not in (
                labels.MIXER_LINEAR,
                labels.MIXER_ANDERSON,
                labels.MIXER_BROYDEN,
            ):
                mixer_labels = (
                    labels.MIXER_LINEAR_STRS
                    + labels.MIXER_ANDERSON_STRS
                    + labels.MIXER_BROYDEN_STRS
                )
                raise ValueError(
                    f"Unknown mixer '{mixer}'. Choose from "
                    f"'{', '.join(mixer_labels)}'."
                )

            self.mixer = mixer
        else:
            raise TypeError(
                "The mixer must be of type 'int' or 'str', but "
                f"'{type(mixer)}' was given."
            )

        self.maxiter = maxiter
        self.damp = damp
        self.force_convergence = force_convergence
        self.batch_mode = batch_mode

        self.device = device
        self.dtype = dtype

        self.x_atol = check_tols(x_atol, dtype)
        self.f_atol = check_tols(f_atol, dtype)

        self.fermi = ConfigFermi(
            etemp=fermi_etemp,
            maxiter=fermi_maxiter,
            thresh=fermi_thresh,
            partition=fermi_partition,
            device=device,
            dtype=dtype,
        )

    def info(self) -> dict[str, Any]:
        """
        Return a dictionary with the SCF configuration.

        Returns
        -------
        dict[str, Any]
            Dictionary with the SCF configuration.
        """
        return {
            "SCF Options": {
                "Guess Method": labels.GUESS_MAP[self.guess],
                "SCF Mode": labels.SCF_MODE_MAP[self.scf_mode],
                "SCP Mode": labels.SCP_MODE_MAP[self.scp_mode],
                "Maxiter": self.maxiter,
                "Mixer": labels.MIXER_MAP[self.mixer],
                "Damping Factor": self.damp,
                "Force Convergence": self.force_convergence,
                "x tolerance": self.x_atol,
                "f(x) tolerance": self.f_atol,
                **self.fermi.info(),
            }
        }

    def __str__(self):
        config_str = [
            f"Configuration for SCF:",
            f"  Guess Method: {self.guess}",
            f"  SCF Mode: {self.scf_mode} (Convergence approach)",
            f"  SCP Mode: {self.scp_mode} (Convergence target)",
            f"  Maximum Iterations: {self.maxiter}",
            f"  Mixer: {self.mixer}",
            f"  Damping Factor: {self.damp}",
            f"  Force Convergence: {self.force_convergence}",
            f"  Device: {self.device}",
            f"  Data Type: {self.dtype}",
            f"  xitorch absolute Tolerance: {self.x_atol}",
            f"  xitorch Functional Tolerance: {self.f_atol}",
            f"  Fermi Configuration: {self.fermi}",
        ]
        return "\n".join(config_str)

    def __repr__(self) -> str:
        return str(self)


def check_tols(value: float, dtype: torch.dtype) -> float:
    """
    Set tolerances to catch unreasonably small values.

    Parameters
    ----------
    value : float
        Selected tolerance that will be checked.
    dtype : torch.dtype
        Floating point precision to adjust tolerances to.

    Returns
    -------
    float
        Possibly corrected tolerance.
    """
    eps = torch.finfo(dtype).eps

    if value < eps:
        OutputHandler.warn(
            f"Selected tolerance ({value:.2E}) is smaller than the "
            f"smallest value for the selected dtype ({dtype}, "
            f"{eps:.2E}). Switching to {100*eps:.2E} instead."
        )
        return 100 * eps

    return value


class ConfigFermi:
    """
    Configuration for fermi smearing.
    """

    etemp: float | int
    """Electronic temperature (in a.u.) for Fermi smearing."""

    maxiter: int
    """Maximum number of iterations for Fermi smearing."""

    thresh: dict
    """Float data type dependent threshold for Fermi iterations."""

    partition: int
    """Partitioning scheme for electronic free energy."""

    # PyTorch

    device: torch.device
    """Device for calculations."""

    dtype: torch.dtype
    """Data type for calculations."""

    def __init__(
        self,
        *,
        etemp: float | int = defaults.FERMI_ETEMP,
        maxiter: int = defaults.FERMI_MAXITER,
        thresh: dict = defaults.FERMI_THRESH,
        partition: str | int = defaults.FERMI_PARTITION,
        # PyTorch
        device: torch.device = get_default_device(),
        dtype: torch.dtype = get_default_dtype(),
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.etemp = etemp
        self.maxiter = maxiter
        self.thresh = thresh

        if isinstance(partition, str):
            if partition.casefold() in labels.FERMI_PARTITION_EQUAL_STRS:
                self.partition = labels.FERMI_PARTITION_EQUAL
            elif partition.casefold() in labels.FERMI_PARTITION_ATOMIC_STRS:
                self.partition = labels.FERMI_PARTITION_ATOMIC
            else:
                fermi_partition_labels = (
                    labels.FERMI_PARTITION_EQUAL_STRS
                    + labels.FERMI_PARTITION_ATOMIC_STRS
                )
                raise ValueError(
                    "Unknown partitioning scheme for the free energy in Fermi "
                    f"smearing '{partition}'. Use one of "
                    f"'{', '.join(fermi_partition_labels)}'."
                )
        elif isinstance(partition, int):
            if partition not in (
                labels.FERMI_PARTITION_EQUAL,
                labels.FERMI_PARTITION_ATOMIC,
            ):
                fermi_partition_labels = (
                    labels.FERMI_PARTITION_EQUAL_STRS
                    + labels.FERMI_PARTITION_ATOMIC_STRS
                )
                raise ValueError(
                    "Unknown partitioning scheme for the free energy in Fermi "
                    f"smearing '{partition}'. Use one of "
                    f"'{', '.join(fermi_partition_labels)}'."
                )

            self.partition = partition
        else:
            raise TypeError(
                "The partition must be of type 'int' or 'str', but "
                f"'{type(partition)}' was given."
            )

    def info(self) -> dict[str, dict[str, float | int | str]]:
        """
        Return a dictionary with the Fermi smearing configuration.

        Returns
        -------
        dict[str, dict[str, float | int | str]]
            Dictionary with the Fermi smearing configuration.
        """
        return {
            "Fermi Smearing": {
                "Temperature": self.etemp,
                "Maxiter": self.maxiter,
                "Threshold": self.thresh[self.dtype].item(),
                "Partioning": labels.FERMI_PARTITION_MAP[self.partition],
            }
        }

    def __str__(self) -> str:
        info = self.info()["Fermi Smearing"]
        info_str = ", ".join(f"{key}={value}" for key, value in info.items())
        return f"{self.__class__.__name__}({info_str})"

    def __repr__(self) -> str:
        return str(self)
