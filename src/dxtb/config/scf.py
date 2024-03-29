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
from tad_mctc.typing import Any, get_default_device, get_default_dtype

from ..constants import defaults, labels
from ..io import OutputHandler


class ConfigSCF:
    """
    Configuration for the SCF.
    """

    # TODO: list all
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

    def __init__(
        self,
        *,
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        mixer: str = defaults.MIXER,
        damp: float = defaults.DAMP,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        x_atol: float = defaults.X_ATOL,
        f_atol: float = defaults.F_ATOL,
        force_convergence: bool = defaults.SCF_FORCE_CONVERGENCE,
        # Fermi
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: dict = defaults.FERMI_THRESH,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
        # PyTorch
        device: torch.device = get_default_device(),
        dtype: torch.dtype = get_default_dtype(),
    ) -> None:
        if isinstance(guess, str):
            if guess.casefold() in labels.GUESS_EEQ_STRS:
                self.guess = labels.GUESS_EEQ
            elif guess.casefold() in labels.GUESS_SAD_STRS:
                self.guess = labels.GUESS_SAD
            else:
                raise ValueError(f"Unknown guess method '{guess}'.")
        elif isinstance(guess, int):
            if guess not in (labels.GUESS_EEQ, labels.GUESS_SAD):
                raise ValueError(f"Unknown guess method '{guess}'.")

            self.guess = guess
        else:
            raise TypeError(
                "The guess must be of type 'int' or 'str', but "
                f"'{type(guess)}' was given."
            )

        if isinstance(scf_mode, str):
            if scf_mode.casefold() in labels.SCF_MODE_IMPLICIT_STRS:
                self.scf_mode = labels.SCF_MODE_IMPLICIT
            elif scf_mode.casefold() in labels.SCF_MODE_FULL_STRS:
                self.scf_mode = labels.SCF_MODE_FULL
            elif scf_mode.casefold() == labels.SCF_MODE_EXPERIMENTAL_STRS:
                self.scf_mode = labels.SCF_MODE_EXPERIMENTAL
            else:
                raise ValueError(f"Unknown SCF mode '{scf_mode}'.")
        elif isinstance(scf_mode, int):
            if scf_mode not in (labels.GUESS_EEQ, labels.GUESS_SAD):
                raise ValueError(f"Unknown SCF mode '{scf_mode}'.")

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
                raise ValueError(f"Unknown convergence target (SCP mode) '{scp_mode}'.")
        elif isinstance(scp_mode, int):
            if scp_mode not in (labels.GUESS_EEQ, labels.GUESS_SAD):
                raise ValueError(f"Unknown convergence target (SCP mode) '{scp_mode}'.")

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
                raise ValueError(
                    f"Unknown mixer '{mixer}'. Choose from "
                    f"'{', '.join(labels.MIXER_MAP)}'."
                )
        elif isinstance(mixer, int):
            if mixer not in (
                labels.MIXER_LINEAR,
                labels.MIXER_ANDERSON,
                labels.MIXER_BROYDEN,
            ):
                raise ValueError(
                    f"Unknown mixer '{mixer}'. Choose from "
                    f"'{', '.join(labels.MIXER_MAP)}'."
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
            f"{eps:.2E}). Switching to 10*{eps:.2E} instead."
        )
        return 10 * eps

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
                raise ValueError(
                    "Unknown partitioning scheme for the free energy in Fermi "
                    f"smearing '{partition}'."
                )
        elif isinstance(partition, int):
            if partition not in (labels.GUESS_EEQ, labels.GUESS_SAD):
                raise ValueError(
                    "Unknown partitioning scheme for the free energy in Fermi "
                    f"smearing '{partition}'."
                )

            self.partition = partition
        else:
            raise TypeError(
                "The partition must be of type 'int' or 'str', but "
                f"'{type(partition)}' was given."
            )

    def info(self) -> dict[str, Any]:
        return {
            "Fermi Smearing": {
                "Temperature": self.etemp,
                "Maxiter": self.maxiter,
                "Threshold": self.thresh[self.dtype].item(),
                "Partioning": labels.FERMI_PARTITION_MAP[self.partition],
            }
        }
