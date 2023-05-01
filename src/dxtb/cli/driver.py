"""
Driver class for running dxtb.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from .. import io
from ..constants import defaults
from ..utils import Timers
from ..xtb import Calculator, Result

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


class Driver:
    """
    Driver class for running dxtb.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.base = self._set_base()
        self.chrg = self._set_attr("chrg")
        self.spin = self._set_attr("spin")

    def _set_base(self) -> Path:
        """
        Set the base path from `args.file`.

        Returns
        -------
        Path
            Parent directory of given coordinate file.
        """
        return Path(self.args.file).resolve().parent

    def _set_attr(self, attr: str) -> int:
        # set charge to input from Namespace
        val = getattr(self.args, attr)

        # only search for file if not specified
        if val is None:
            # use charge (or spin) from file or set to zero
            if Path(self.base, FILES[attr]).is_file():
                val = io.read_chrg(Path(self.base, FILES[attr]))
            else:
                val = 0

        return val

    def singlepoint(self) -> Result:
        args = self.args

        if args.detect_anomaly:
            # pylint: disable=import-outside-toplevel
            from torch.autograd.anomaly_mode import set_detect_anomaly

            set_detect_anomaly(True)

        timer = Timers()
        timer.start("setup")

        dd = {"device": args.device, "dtype": args.dtype}

        opts = {
            "etemp": args.etemp,
            "maxiter": args.maxiter,
            "spin": args.spin,
            "verbosity": args.verbosity,
            "exclude": args.exclude,
            "xitorch_xatol": defaults.XITORCH_XATOL,
            "xitorch_fatol": defaults.XITORCH_XATOL,
        }

        _numbers, _positions = io.read_structure_from_file(args.file)
        numbers = torch.tensor(_numbers, dtype=torch.long, device=dd["device"])
        positions = torch.tensor(_positions, **dd)
        chrg = torch.tensor(self.chrg, **dd)

        if args.grad is True:
            positions.requires_grad = True

        # METHOD
        if args.method.lower() == "gfn1" or args.method.lower() == "gfn1-xtb":
            # pylint: disable=import-outside-toplevel
            from ..param import GFN1_XTB as par
        elif args.method.lower() == "gfn2" or args.method.lower() == "gfn2-xtb":
            raise NotImplementedError("GFN2-xTB is not implemented yet.")
        else:
            raise ValueError(f"Unknown method '{args.method}'.")

        # GUESS
        if args.guess.lower() == "eeq" or args.guess.lower() == "sad":
            opts["guess"] = args.guess.lower()
        else:
            raise ValueError(f"Unknown guess method '{args.guess}'.")

        # setup calculator
        timer.stop("setup")
        calc = Calculator(numbers, par, opts=opts, timer=timer, **dd)

        # run singlepoint calculation
        result = calc.singlepoint(numbers, positions, chrg, grad=args.grad)

        if args.verbosity > 0:
            timer.print_times()

        return result

    def __repr__(self) -> str:  # pragma: no cover
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.args})"
