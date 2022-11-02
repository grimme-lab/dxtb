"""
Driver class for running dxtb.
"""

from argparse import Namespace
from pathlib import Path

import torch

from ..io import read_chrg, read_coord
from ..utils import Timers
from ..xtb import Calculator

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


def read_mol(file):

    geo = read_coord(file)
    assert len(geo[0]) == 4
    positions = [g[:3] for g in geo]
    numbers = [g[-1] for g in geo]

    return numbers, positions


class Driver:
    """
    Driver class for running dxtb.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.base = self._set_base()
        self.chrg = self._set_chrg_spin("chrg")
        self.spin = self._set_chrg_spin("spin")

    def _set_base(self) -> Path:
        """
        Set the base path from `args.file`.

        Returns
        -------
        Path
            Parent directory of given coordinate file.
        """
        return Path(self.args.file).resolve().parent

    def _set_attr(self, attr: str):

        # set charge to input from Namespace
        val = getattr(self.args, attr)

        # only search for file if not specified
        if val is None:
            # use charge (or spin) from file or set to zero
            if Path(self.base, FILES[attr]).is_file():
                val = read_chrg(Path(self.base, FILES[attr]))
            else:
                val = 0

        return val

    def singlepoint(self) -> None:
        args = self.args

        timer = Timers()
        timer.start("total")
        timer.start("setup")

        opts = {
            "etemp": args.etemp,
            "verbosity": args.verbosity,
            "maxiter": args.maxiter,
        }

        numbers, positions = read_mol(args.file)

        numbers = torch.tensor(numbers)
        positions = torch.tensor(positions)
        chrg = torch.tensor(self.chrg)

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
        calc = Calculator(numbers, positions, par)
        timer.stop("setup")

        # run singlepoint calculation
        timer.start("singlepoint")
        result = calc.singlepoint(numbers, positions, chrg, opts)
        total = result.total.sum(-1)
        timer.stop("singlepoint")

        # gradient of total energy w.r.t. positions
        if args.grad is True:
            timer.start("grad")
            total.backward()
            if positions.grad is None:
                raise RuntimeError("No gradients found for positions.")
            timer.stop("grad")

        # print results
        timer.stop("total")
        timer.print_times("")

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}(chrg={self.chrg}, spin={self.spin})"
