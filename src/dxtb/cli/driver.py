"""
Driver class for running dxtb.
"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch

from .. import io
from ..constants import defaults, units
from ..interaction.external import new_efield
from ..utils import Timers, batch
from ..xtb import Calculator, Result

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


def print_grad(grad, numbers) -> None:
    from ..constants import PSE

    print("************************Gradient************************")
    print("")

    # Iterate over the tensor and print
    for i, row in enumerate(grad):
        # Get the atomic symbol corresponding to the atomic number
        symbol = PSE.get(int(numbers[i].item()), "?")
        print(f"{i+1:>3} {symbol:<2} : {row[0]:>15.9f} {row[1]:>15.9f} {row[2]:>15.9f}")


class Driver:
    """
    Driver class for running dxtb.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.batched = True
        self.base = self._set_base()
        self.chrg = self._set_attr("chrg")
        self.spin = self._set_attr("spin")

    def _set_base(self) -> list[Path]:
        """
        Set the base path from `args.file`.

        Returns
        -------
        Path
            Parent directory of given coordinate file.
        """
        return [Path(f).resolve().parent for f in self.args.file]

    def _set_attr(self, attr: str) -> int | list[int]:
        # set charge to input from Namespace
        val = getattr(self.args, attr)

        if isinstance(val, list):
            raise NotImplementedError("Setting charges as list not working.")

        # only search for file if not specified
        if val is not None:
            return val

        vals = []
        for path in self.base:
            # use charge (or spin) from file or set to zero
            if Path(path, FILES[attr]).is_file():
                vals.append(io.read_chrg(Path(path, FILES[attr])))
            else:
                vals.append(0)

        # do not return list if only one system is considered
        if len(vals) == 1:
            vals = vals[0]

        return vals

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
            "damp": args.damp,
            "scf_mode": args.scf_mode,
            "scp_mode": args.scp_mode,
            "int_driver": args.int_driver,
            "mixer": args.mixer,
            "maxiter": args.maxiter,
            "spin": args.spin,
            "verbosity": args.verbosity,
            "exclude": args.exclude,
            "xitorch_xatol": args.xtol,
            "xitorch_fatol": args.ftol,
        }

        if len(self.args.file) > 1:
            _n = []
            _p = []
            for f in self.args.file:
                n, p = io.read_structure_from_file(f)
                _n.append(torch.tensor(n, dtype=torch.long, device=dd["device"]))
                _p.append(torch.tensor(p, **dd))
            numbers = batch.pack(_n)
            positions = batch.pack(_p)
        else:
            _n, _p = io.read_structure_from_file(args.file[0])
            numbers = torch.tensor(_n, dtype=torch.long, device=dd["device"])
            positions = torch.tensor(_p, **dd)

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

        # INTERACTIONS
        interactions = []
        if args.efield is not None or args.ir is True or args.raman is True:
            field = (
                torch.tensor(
                    args.efield if args.efield is not None else [0, 0, 0], **dd
                )
                * units.VAA2AU
            )
            interactions.append(new_efield(field, **dd))

        # setup calculator
        timer.stop("setup")
        calc = Calculator(
            numbers, par, opts=opts, interaction=interactions, timer=timer, **dd
        )

        # run singlepoint calculation
        result = calc.singlepoint(numbers, positions, chrg)

        ####################################################
        if args.grad:
            timer.start("grad")
            (g,) = torch.autograd.grad(result.total.sum(), positions)
            timer.stop("grad")
        #####################################################

        if args.grad and args.verbosity > 2:
            print_grad(result.total_grad.clone(), numbers)
            print("")

        if args.ir is True:
            timer.start("IR")
            calc.ir_spectrum_num(numbers, positions, chrg)
            timer.stop("IR")

        if args.verbosity > 0:
            timer.print_times()

        return result

    def __repr__(self) -> str:  # pragma: no cover
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.args})"
