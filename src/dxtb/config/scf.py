"""
SCF configuration.
"""
from __future__ import annotations

import torch

from ..constants import defaults, labels
from ..io import OutputHandler


class ConfigSCF:
    """
    Configuration for the SCF.
    """

    # TODO: list all

    scf_mode: int
    """SCF Convergence approach (denoted by backward strategy)."""

    scp_mode: int
    """Convergence target (self-consistent property)."""

    def __init__(
        self,
        *,
        guess=defaults.GUESS,
        maxiter=defaults.MAXITER,
        mixer=defaults.MIXER,
        damp=defaults.DAMP,
        scf_mode=defaults.SCF_MODE,
        scp_mode=defaults.SCP_MODE,
        xatol=defaults.XITORCH_XATOL,
        fatol=defaults.XITORCH_FATOL,
        force_convergence=defaults.SCF_FORCE_CONVERGENCE,
        # Fermi
        fermi_etemp=defaults.FERMI_ETEMP,
        fermi_maxiter=defaults.FERMI_MAXITER,
        fermi_thresh=defaults.FERMI_THRESH,
        fermi_partition=defaults.FERMI_PARTITION,
        # PyTorch
        device=defaults.get_default_device(),
        dtype=defaults.get_default_dtype(),
    ) -> None:
        if guess.casefold() in ("eeq", "equilibration"):
            self.guess = labels.GUESS_EEQ
        elif guess.casefold() in ("sad", "zero"):
            self.guess = labels.GUESS_SAD
        else:
            raise ValueError(f"Unknown guess method '{guess}'.")

        if scf_mode.casefold() in ("default", "implicit"):
            self.scf_mode = labels.SCF_MODE_IMPLICIT
        elif scf_mode.casefold() in ("full", "full_tracking"):
            self.scf_mode = labels.SCF_MODE_FULL
        elif scf_mode.casefold() == ("experimental", "perfect"):
            self.scf_mode = labels.SCF_MODE_EXPERIMENTAL
        else:
            raise ValueError(f"Unknown SCF mode '{scf_mode}'.")

        if scp_mode.casefold() in ("charge", "charges"):
            self.scp_mode = labels.SCP_MODE_CHARGE
        elif scp_mode.casefold() in ("potential", "pot"):
            self.scp_mode = labels.SCP_MODE_POTENTIAL
        elif scp_mode.casefold() in ("fock", "fockian"):
            self.scp_mode = labels.SCP_MODE_FOCK
        else:
            raise ValueError(f"Unknown convergence target (SCP mode) '{scp_mode}'.")

        self.maxiter = maxiter
        self.mixer = mixer
        self.damp = damp
        self.force_convergence = force_convergence

        self.device = device
        self.dtype = dtype

        self.xatol = check_tols(xatol, dtype)
        self.fatol = check_tols(fatol, dtype)

        self.fermi = ConfigFermi(
            etemp=fermi_etemp,
            maxiter=fermi_maxiter,
            thresh=fermi_thresh,
            partition=fermi_partition,
        )


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

    def __init__(
        self,
        *,
        etemp=defaults.FERMI_ETEMP,
        maxiter=defaults.FERMI_MAXITER,
        thresh=defaults.FERMI_THRESH,
        partition=defaults.FERMI_PARTITION,
    ) -> None:
        self.etemp = etemp
        self.maxiter = maxiter
        self.thresh = thresh

        if partition.casefold() in ("equal", "same"):
            self.partition = labels.FERMI_PARTITION_EQUAL
        elif partition.casefold() in ("atom", "atomic"):
            self.partition = labels.FERMI_PARTITION_ATOMIC
        else:
            raise ValueError(
                "Unknown partitioning scheme for the free energy in Fermi "
                f"smearing '{partition}'."
            )
