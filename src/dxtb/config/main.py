from __future__ import annotations

import sys
from argparse import Namespace

import torch

from .._types import Self
from ..constants import defaults, labels
from .integral import ConfigIntegrals
from .scf import ConfigSCF


class Config:
    """
    Configuration of the calculation.
    """

    def __init__(
        self,
        *,
        file=None,
        exclude: str | list[str] = defaults.EXCLUDE,
        method: str | int = defaults.METHOD,
        grad: bool = False,
        # integrals
        int_cutoff: float = defaults.INTCUTOFF,
        int_driver: str | int = defaults.INTDRIVER,
        int_uplo: str = defaults.INTUPLO,
        # PyTorch
        device: torch.device = defaults.get_default_device(),
        dtype: torch.dtype = defaults.get_default_dtype(),
        anomaly: bool = False,
        # SCF
        guess: str | int = defaults.GUESS,
        maxiter: int = defaults.MAXITER,
        mixer: str = defaults.MIXER,
        damp: float = defaults.DAMP,
        scf_mode: str | int = defaults.SCF_MODE,
        scp_mode: str | int = defaults.SCP_MODE,
        xitorch_xatol: float = defaults.XITORCH_XATOL,
        xitorch_fatol: float = defaults.XITORCH_FATOL,
        force_convergence: bool = False,
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: dict = defaults.FERMI_THRESH,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
    ) -> None:
        self.file = file
        self.exclude = exclude
        self.grad = grad

        self.device = device
        self.dtype = dtype
        self.anomaly = anomaly

        if isinstance(method, str):
            if method.casefold() in labels.GFN1_XTB_STRS:
                self.method = labels.GFN1_XTB
            elif method.casefold() in labels.GFN2_XTB_STRS:
                self.method = labels.GFN2_XTB
            else:
                raise ValueError(f"Unknown xtb method '{method}'.")
        elif isinstance(method, int):
            if method not in (labels.GFN1_XTB, labels.GFN2_XTB):
                raise ValueError(f"Unknown xtb method '{method}'.")

            self.method = method
        else:
            raise TypeError(
                "The method must be of type 'int' or 'str', but "
                f"'{type(method)}' was given."
            )

        self.ints = ConfigIntegrals(
            cutoff=int_cutoff,
            driver=int_driver,
            uplo=int_uplo,
        )

        self.scf = ConfigSCF(
            guess=guess,
            maxiter=maxiter,
            mixer=mixer,
            damp=damp,
            scf_mode=scf_mode,
            scp_mode=scp_mode,
            xitorch_xatol=xitorch_xatol,
            xitorch_fatol=xitorch_fatol,
            force_convergence=force_convergence,
            fermi_etemp=fermi_etemp,
            fermi_maxiter=fermi_maxiter,
            fermi_thresh=fermi_thresh,
            fermi_partition=fermi_partition,
        )

        # compatibility checks
        if (
            self.method == labels.GFN2_XTB
            and self.ints.driver != labels.INTDRIVER_LIBCINT
        ):
            raise RuntimeError(
                "Multipole integrals not available in PyTorch integral drivers."
                " Use `libcint` as backend."
            )

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        return cls(
            # general
            file=args.file,
            exclude=args.exclude,
            int_driver=args.int_driver,
            method=args.method,
            grad=args.grad,
            # PyTorch
            device=args.device,
            dtype=args.dtype,
            anomaly=args.detect_anomaly,
            # SCF
            guess=args.guess,
            maxiter=args.maxiter,
            mixer=args.mixer,
            damp=args.damp,
            scf_mode=args.scf_mode,
            scp_mode=args.scp_mode,
            xitorch_xatol=args.xtol,
            xitorch_fatol=args.ftol,
            fermi_etemp=args.etemp,
        )

    def info(self) -> dict:
        return {
            "Calculation Configuration": {
                "Program Call": " ".join(sys.argv),
                "Input File": self.file,
                "Method": labels.GFN_XTB_MAP[self.method],
                "Excluded": None if len(self.exclude) == 0 else self.exclude,
                "Gradient": self.grad,
                "Integral driver": labels.INTDRIVER_MAP[self.ints.driver],
                "FP accuracy": str(self.dtype),
                "Device": str(self.device),
            },
            **self.scf.info(),
        }

    def __str__(self) -> str:
        return f"{self.device}"
