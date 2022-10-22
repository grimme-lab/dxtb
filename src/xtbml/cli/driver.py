"""
Driver class for running dxtb.
"""

from argparse import Namespace

from pathlib import Path

import torch

from ..io import read_chrg, read_coord
from ..utils import Timers
from ..xtb import Calculator

FILES = {"spin": ".UHF", "charge": ".CHRG"}


def read_mol(args):
    FNAME = "coord"

    base = Path(args.file).resolve().parent
    if Path(base, FNAME).is_file():
        geo = read_coord(Path(base, FNAME))
    else:
        raise FileNotFoundError(f"'{FNAME}' file not found in '{base}'")

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
        self.chrg = self._set_total_chrg()
        self.spin = self._set_spin()

    def _set_total_chrg(self):

        # set charge to input
        chrg = self.args.chrg

        # only search for .CHRG file if not specified
        if chrg is None:
            base = Path(self.args.file).resolve().parent

            # use charge from file or set to zero
            if Path(base, FILES["charge"]).is_file():
                chrg = read_chrg(Path(base, FILES["charge"]))
            else:
                chrg = 0

        return chrg

    def _set_spin(self):

        # set charge to input
        spin = self.args.spin

        # only search for .CHRG file if not specified
        if spin is None:
            base = Path(self.args.file).resolve().parent

            # use charge from file or set to zero
            if Path(base, FILES["spin"]).is_file():
                spin = read_chrg(Path(base, FILES["spin"]))
            else:
                spin = 0

        return spin

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

        if args.method.lower() == "gfn1" or args.method.lower() == "gfn1-xtb":
            # pylint: disable=import-outside-toplevel
            from ..param import GFN1_XTB as par
        elif args.method.lower() == "gfn2" or args.method.lower() == "gfn2-xtb":
            raise NotImplementedError("GFN2-xTB is not implemented yet.")
        else:
            raise ValueError(f"Unknown method: {args.method}")

        # setup calculator
        calc = Calculator(numbers, positions, par)
        timer.stop("setup")

        # run singlepoint calculation
        timer.start("singlepoint")

        # GUESS
        if args.guess.lower() == "eeq" or args.guess.lower() == "sad":
            opts["guess"] = args.guess.lower()
        else:
            raise ValueError(f"Unknown guess method '{args.guess}'.")

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
        # timer.print_times()
        torch.set_printoptions(precision=16)
        print(total)
        print(result.scf.sum(-1))
