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
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        mixer: str = defaults.MIXER,
        damp: float = defaults.DAMP,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        xitorch_xatol: float = defaults.XITORCH_XATOL,
        xitorch_fatol: float = defaults.XITORCH_FATOL,
        force_convergence=defaults.SCF_FORCE_CONVERGENCE,
        # Fermi
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: dict = defaults.FERMI_THRESH,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
        # PyTorch
        device: torch.device = defaults.get_default_device(),
        dtype: torch.dtype = defaults.get_default_dtype(),
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

        self.maxiter = maxiter
        self.mixer = mixer
        self.damp = damp
        self.force_convergence = force_convergence

        self.device = device
        self.dtype = dtype

        self.xitorch_xatol = check_tols(xitorch_xatol, dtype)
        self.xitorch_fatol = check_tols(xitorch_fatol, dtype)

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
    ) -> None:
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
