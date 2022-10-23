"""
Parser for command line options.
"""

import argparse
from pathlib import Path


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


class EtempAction(argparse.Action):
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


def argparser(name: str = "dxtb") -> argparse.ArgumentParser:
    """
    Parses the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Container for command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="dxtb - Fully differentiable extended tight-binding.",
        prog=name,
        formatter_class=lambda prog: Formatter(prog, max_help_position=60),
    )

    parser.add_argument(
        "--version",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Show version.",
    )
    parser.add_argument(
        "--chrg",
        type=int,
        help="R|Molecular charge.",
    )
    parser.add_argument(
        "--spin",
        "--uhf",
        type=int,
        default=0,
        help="R|Molecular spin.",
    )
    parser.add_argument(
        "--etemp",
        action=EtempAction,
        type=float,
        default=300.0,
        help="R|Electronic Temperature in K.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="R|Maximum number of SCF iterations.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        help="R|Verbosity level of printout.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gfn1",
        help="R|Method for calculation. (Options: 'gfn1', 'gfn1-xtb')",
    )
    parser.add_argument(
        "--guess",
        type=str,
        default="eeq",
        help="R|Model for initial charges. (Options: 'sad', 'eeq')",
    )
    parser.add_argument(
        "--grad",
        action="store_true",
        help="R|Whether to compute gradients for positions w.r.t. energy.",
    )
    parser.add_argument(
        "file",
        nargs="?",
        type=is_file,  # manual validation
        help="R|Path to coordinate file.",
    )

    return parser
