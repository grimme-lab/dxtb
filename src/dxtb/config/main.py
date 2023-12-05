from argparse import Namespace

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
        exclude=defaults.EXCLUDE,
        method=defaults.METHOD,
        grad=False,
        # integrals
        int_cutoff=defaults.INTCUTOFF,
        int_driver=defaults.INTDRIVER,
        int_uplo=defaults.INTUPLO,
        # PyTorch
        device=defaults.TORCH_DEVICE,
        dtype=defaults.TORCH_DTYPE,
        anomaly=False,
        # SCF
        guess=defaults.GUESS,
        maxiter=defaults.MAXITER,
        mixer=defaults.MIXER,
        damp=defaults.DAMP,
        scf_mode=defaults.SCF_MODE,
        scp_mode=defaults.SCP_MODE,
        xatol=defaults.XITORCH_XATOL,
        fatol=defaults.XITORCH_FATOL,
        fermi_etemp=defaults.FERMI_ETEMP,
        fermi_maxiter=defaults.FERMI_MAXITER,
        fermi_thresh=defaults.FERMI_THRESH,
        fermi_partition=defaults.FERMI_PARTITION,
    ) -> None:
        self.file = file
        self.exclude = exclude
        self.grad = grad

        self.device = device
        self.dtype = dtype
        self.anomaly = anomaly

        if method.casefold() in ("gfn1", "gfn1-xtb", "gfn1_xtb", "gfn1xtb"):
            self.method = labels.GFN1_XTB
        elif method.casefold() in ("gfn2", "gfn2-xtb", "gfn2_xtb", "gfn2xtb"):
            self.method = labels.GFN2_XTB
        else:
            raise ValueError(f"Unknown xtb method '{method}'.")

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
            xatol=xatol,
            fatol=fatol,
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
            xatol=args.xtol,
            fatol=args.ftol,
            fermi_etemp=args.etemp,
        )
