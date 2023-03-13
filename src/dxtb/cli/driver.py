"""
Driver class for running dxtb.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from .. import io
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

        timer = Timers()
        timer.start("setup")

        dd = {"device": args.device, "dtype": args.dtype}

        opts = {
            "etemp": args.etemp,
            "maxiter": args.maxiter,
            "spin": args.spin,
            "verbosity": args.verbosity,
            "exclude": args.exclude,
            "xitorch_xatol": 1e-6,
            "xitorch_fatol": 1e-6,
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
        calc = Calculator(numbers, par, opts=opts, **dd)
        timer.stop("setup")

        # run singlepoint calculation, timer set within
        result = calc.singlepoint(numbers, positions, chrg, timer=timer, grad=args.grad)

        # gradient of total energy w.r.t. positions
        # if args.grad is True:
        #     timer.start("grad")
        #     total = result.total.sum(-1)
        #     total.backward()
        #     if positions.grad is None:
        #         raise RuntimeError("No gradients found for positions.")
        #     result.gradient = positions.grad
        #     timer.stop("grad")

        # stop timer
        timer.stop("total")
        result.timer = timer

        if args.verbosity > 0:
            timer.print_times()

        return result

    def __str__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.args})"
