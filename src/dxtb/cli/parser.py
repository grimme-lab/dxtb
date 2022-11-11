"""
Parser for command line options.
"""

import argparse
from pathlib import Path

import torch

from ..constants import defaults


def is_file(path: str | Path) -> str | Path:
    p = Path(path)
    if p.is_dir():
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': Is a directory.",
        )

    if p.is_file() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': No such file.",
        )

    return path


def is_dir(path: str | Path) -> str | Path:
    p = Path(path)

    if p.is_file() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': No such file or directory.",
        )

    if p.is_dir() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': Is not a directory.",
        )

    return path


def action_not_less_than(min_value: float = 0.0):
    class CustomAction(argparse.Action):
        """
        Custom action for limiting possible input values.
        """

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: list[float | int] | float | int,
            option_string: str | None = None,
        ) -> None:
            if isinstance(values, (int, float)):
                values = [values]

            if any(value < min_value for value in values):
                parser.error(
                    f"Option '{option_string}' takes only positive values ({values})."
                )

            if len(values) == 1:
                values = values[0]

            setattr(args, self.dest, values)

    return CustomAction


class ConvertToTorchDtype(argparse.Action):
    """
    Custom action for converting an input value string to a PyTorch dtype.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        values: str | torch.dtype,
        option_string: str | None = None,
    ) -> None:
        match values:
            case "float16":
                values = torch.float16
            case "float32" | "sp":
                values = torch.float32
            case "float64" | "double" | "dp":
                values = torch.float64
            case _:
                parser.error(
                    f"Option '{option_string}' was passed an unknown keyword ({values})."
                )

        setattr(args, self.dest, values)


class ConvertToTorchDevice(argparse.Action):
    """
    Custom action for converting an input string to a PyTorch device.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        allowed_devices = ("cpu", "cuda")
        err_msg = (
            f"Option '{option_string}' was passed an unknown keyword ({values})."
            "Use 'cpu', 'cpu:<INTEGER>', 'cuda', or 'cuda:<INTEGER>'."
        )

        if values == "cpu":
            setattr(args, self.dest, torch.device(values))
            return

        if values == "cuda":
            dev = f"{values}:{torch.cuda.current_device()}"
            setattr(args, self.dest, torch.device(dev))
            return

        if ":" in values:
            dev, idx = values.split(":")
            if dev not in allowed_devices:
                parser.error(err_msg)

            if idx.isdigit() is False:
                parser.error(err_msg)

            setattr(args, self.dest, torch.device(values))
            return

        parser.error(err_msg)


class ActionNonNegative(argparse.Action):
    """
    Custom action for limiting possible input values for the electronic temperature.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        values: float,
        option_string: str | None = None,
    ) -> None:
        if values < 0.0:
            parser.error(
                f"Electronic Temperature ({option_string}) must be positive or None ({values})."
            )

        setattr(args, self.dest, values)


class Formatter(argparse.HelpFormatter):
    """
    Custom format for help message.
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        """
        Append default value and type of action to help string.

        Parameters
        ----------
        action : argparse.Action
            Command line option.

        Returns
        -------
        str | None
            Help string.
        """
        helper = action.help
        if helper is not None and "%(default)" not in helper:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]

                if action.option_strings or action.nargs in defaulting_nargs:
                    helper += "\n - default: %(default)s"
                if action.type:
                    helper += "\n - type: %(type)s"

        return helper

    def _split_lines(self, text: str, width: int) -> list[str]:
        """
        Re-implementation of `RawTextHelpFormatter._split_lines` that includes
        line breaks for strings starting with 'R|'.

        Parameters
        ----------
        text : str
            Help message.
        width : int
            Text width.

        Returns
        -------
        list[str]
            Split text.
        """
        if text.startswith("R|"):
            return text[2:].splitlines()

        # pylint: disable=protected-access
        return argparse.HelpFormatter._split_lines(self, text, width)


def argparser(name: str = "dxtb", **kwargs) -> argparse.ArgumentParser:
    """
    Parses the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Container for command line arguments.
    """

    desc = kwargs.pop(
        "description", "dxtb - Fully differentiable extended tight-binding program."
    )

    parser = argparse.ArgumentParser(
        description=desc,
        prog=name,
        formatter_class=lambda prog: Formatter(prog, max_help_position=60),
        add_help=False,
        **kwargs,
    )
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show version and exit.",
    )
    parser.add_argument(
        "--chrg",
        action=action_not_less_than(-10.0),
        type=int,
        default=defaults.CHRG,
        nargs="+",
        help="R|Molecular charge.",
    )
    parser.add_argument(
        "--spin",
        "--uhf",
        action=action_not_less_than(0.0),
        type=int,
        default=defaults.SPIN,
        nargs="+",
        help="R|Molecular spin.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        default=defaults.EXCLUDE,
        choices=defaults.EXCLUDE_CHOICES,
        nargs="+",
        help="R|Turn off energy contributions.",
    )
    parser.add_argument(
        "--etemp",
        action=action_not_less_than(0.0),
        type=float,
        default=defaults.ETEMP,
        help="R|Electronic Temperature in K.",
    )
    parser.add_argument(
        "--dtype",
        action=ConvertToTorchDtype,
        type=str,
        default=defaults.TORCH_DTYPE,
        choices=defaults.TORCH_DTYPE_CHOICES,
        help="R|Data type for PyTorch floating point tensors.",
    )
    parser.add_argument(
        "--device",
        action=ConvertToTorchDevice,
        type=str,
        default=defaults.TORCH_DEVICE,
        help="R|Device for PyTorch tensors.",
    )
    parser.add_argument(
        "--fermi_maxiter",
        type=int,
        default=defaults.FERMI_MAXITER,
        help="R|Maximum number of iterations for Fermi smearing.",
    )
    parser.add_argument(
        "--fermi_energy_partition",
        type=str,
        default=defaults.FERMI_FENERGY_PARTITION,
        choices=defaults.FERMI_FENERGY_PARTITION_CHOICES,
        help="R|Partitioning scheme for electronic free energy.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=defaults.MAXITER,
        help="R|Maximum number of SCF iterations.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=defaults.VERBOSITY,
        help="R|Verbosity level of printout.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=defaults.METHOD,
        choices=defaults.METHOD_CHOICES,
        help="R|Method for calculation.",
    )
    parser.add_argument(
        "--guess",
        type=str,
        default=defaults.GUESS,
        choices=defaults.GUESS_CHOICES,
        help="R|Model for initial charges.",
    )
    parser.add_argument(
        "--grad",
        action="store_true",
        help="R|Whether to compute gradients for positions w.r.t. energy.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "R|Profile the program.\n"
            "Creates 'dxtb.profile' that can be analyzed with 'snakeviz'."
        ),
    )
    parser.add_argument(
        "--xtol",
        type=float,
        default=defaults.XITORCH_XATOL,
        help="R|Set absolute tolerance for SCF (input).",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=defaults.XITORCH_FATOL,
        help="R|Set absolute tolerance for SCF (output).",
    )

    parser.add_argument(
        "--filetype",
        type=str,
        choices=["xyz", "tm", "tmol", "turbomole", "json", "qcschema"],
        help="R|Explicitly set file type of input.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=is_file,  # manual validation
        help="R|Path to coordinate file.",
    )

    return parser
