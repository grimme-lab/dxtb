"""
Parser for command line options.
"""

import argparse


class EtempAction(argparse.Action):
    """
    Custom action for limiting possible input values for the electronic temperature.
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        args: argparse.Namespace,
        values,
        option_string=None,
    ) -> None:
        if values < 0.0:
            parser.error(
                f"Electronic Temperature must be positive or None ({option_string})."
            )

        setattr(args, self.dest, values)


def argparser(name: str = "dxtb") -> argparse.ArgumentParser:
    """
    Parses the command line arguments.

    Returns
    -------
    argparse.ArgumentParser
        Container for command line arguments.
    """

    parser = argparse.ArgumentParser(
        description="dxtb - Fully differentiable xtb",
        prog=name,
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=60),
    )

    parser = argparse.ArgumentParser("dxtb options")
    parser.add_argument(
        "--chrg",
        type=int,
        help="Molecular charge. (Type: int ; Default: 0)",
    )
    parser.add_argument(
        "--spin",
        "--uhf",
        type=int,
        default=0,
        help="Molecular spin. (Type: int ; Default: 0)",
    )
    parser.add_argument(
        "--etemp",
        action=EtempAction,
        type=float,
        default=300.0,
        help="Electronic Temperature in K. (Type: float ; Default: 300)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Maximum number of SCF iterations. (Type: int ; Default: 20)",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        default=1,
        help="Verbosity level of printout. (Type: int ; Default: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gfn1",
        help=(
            "Method for calculation. "
            "(Type: str, case insensitive ; "
            "Default: 'gfn1' ; Options: 'gfn1', 'gfn1-xtb')"
        ),
    )
    parser.add_argument(
        "--guess",
        type=str,
        default="sad",
        help=(
            "Model for initial charges. "
            "(Type: str, case insensitive ; "
            "Default: 'sad' ; Options: 'sad', 'eeq')"
        ),
    )
    parser.add_argument(
        "--grad",
        action="store_true",
        help="Whether to compute gradients for positions w.r.t. energy.",
    )
    parser.add_argument("file")

    return parser
