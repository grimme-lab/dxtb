"""
Parser for command line options.
"""

import argparse
from pathlib import Path

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
            values: float,
            option_string: str | None = None,
        ) -> None:
            if values < min_value:
                parser.error(
                    f"Option '{option_string}' takes only positive values ({values})."
                )

            setattr(args, self.dest, values)

    return CustomAction


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
        type=int,
        default=defaults.CHRG,
        help="R|Molecular charge.",
    )
    parser.add_argument(
        "--spin",
        "--uhf",
        action=action_not_less_than(0.0),
        type=int,
        default=defaults.SPIN,
        help="R|Molecular spin.",
    )
    parser.add_argument(
        "--etemp",
        action=action_not_less_than(0.0),
        type=float,
        default=defaults.ETEMP,
        help="R|Electronic Temperature in K.",
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
        "file",
        nargs="?",
        type=is_file,  # manual validation
        help="R|Path to coordinate file.",
    )

    return parser
