"""
Driver class for running dxtb.
"""

from argparse import Namespace
from pathlib import Path

import torch

from .. import io
from ..utils import Timers
from ..xtb import Calculator

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


def read_structure_from_file(
    file: str, ftype: str | None = None
) -> tuple[list[int], list[list[float]]]:
    """
    Helper to read the structure from the given file.

    Parameters
    ----------
    file : str
        Path of file containing the structure.
    ftype : str | None, optional
        File type. Defaults to `None`, i.e., infered from the extension.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Lists of atoms and coordinates.

    Raises
    ------
    FileNotFoundError
        File given does not exist.
    NotImplementedError
        Reader for specific file type not implemented.
    ValueError
        Unknown file type.
    """

    f = Path(file)
    if f.exists() is False:
        raise FileNotFoundError(f"File '{f}' not found.")

    if ftype is None:
        ftype = f.suffix.lower()[1:]

    match [ftype, f.name.lower()]:
        case ["xyz" | "log", *_]:
            numbers, positions = io.read_xyz(f)
        case ["", "coord"] | ["tmol" | "tm" | "turbomole", *_]:
            numbers, positions = io.read_coord(f)
        case ["mol", *_] | ["sdf", *_] | ["gen", *_] | ["pdb", *_]:
            raise NotImplementedError(
                f"Filetype '{ftype}' recognized but no reader available."
            )
        case ["qchem", *_]:
            raise NotImplementedError(
                f"Filetype '{ftype}' (Q-Chem) recognized but no reader available."
            )
        case ["poscar" | "contcar" | "vasp" | "crystal", *_] | [
            "",
            "poscar" | "contcar" | "vasp",
        ]:
            raise NotImplementedError(
                "VASP/CRYSTAL file recognized but no reader available."
            )
        case ["ein" | "gaussian", *_]:
            raise NotImplementedError(
                f"Filetype '{ftype}' (Gaussian) recognized but no reader available."
            )
        case ["json" | "qcschema", *_]:
            numbers, positions = io.read_qcschema(f)
        case _:
            raise ValueError(f"Unknown filetype '{ftype}' in '{f}'.")

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

    def _set_chrg_spin(self, prop: str):

        # set charge to input from Namespace
        val = getattr(self.args, prop)

        # only search for file if not specified
        if val is None:
            # use charge from file or set to zero
            if Path(self.base, FILES[prop]).is_file():
                val = io.read_chrg(Path(self.base, FILES[prop]))
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

        numbers, positions = read_structure_from_file(args.file)
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
