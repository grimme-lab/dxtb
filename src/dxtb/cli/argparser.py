"""
Parser for command line options.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .. import __version__
from .._types import Any
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

    if p.is_file() is True:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': Is not a directory.",
        )

    if p.is_dir() is False:
        raise argparse.ArgumentTypeError(
            f"Cannot open '{path}': No such file or directory.",
        )

    return path


def action_not_less_than(min_value: float = 0.0):
    class CustomAction(argparse.Action):
        """
        Custom action for limiting possible input values.
        """

        def __call__(
            self,
            p: argparse.ArgumentParser,
            args: argparse.Namespace,
            values: list[float | int] | float | int,
            option_string: str | None = None,
        ) -> None:
            if isinstance(values, (int, float)):
                values = [values]

            if any(value < min_value for value in values):
                p.error(
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
        p: argparse.ArgumentParser,
        args: argparse.Namespace,
        values: str | torch.dtype,
        option_string: str | None = None,
    ) -> None:
        if values in ("float16", torch.float16):
            values = torch.float16
        elif values in ("float32", torch.float32, "sp"):
            values = torch.float32
        elif values in ("float64", torch.float64, "double", torch.double, "dp"):
            values = torch.float64
        # unreachable due to choices
        else:  # pragma: no cover
            p.error(
                f"Option '{option_string}' was passed unknown keyword " f"({values})."
            )

        setattr(args, self.dest, values)


class ConvertToTorchDevice(argparse.Action):
    """
    Custom action for converting an input string to a PyTorch device.
    """

    def __call__(
        self,
        p: argparse.ArgumentParser,
        args: argparse.Namespace,
        values: str,
        option_string: str | None = None,
    ) -> None:
        allowed_devices = ("cpu", "cuda")
        err_msg = (
            f"Option '{option_string}' was passed unknown keyword ({values}). "
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
                p.error(err_msg)

            if idx.isdigit() is False:
                p.error(err_msg)

            setattr(args, self.dest, torch.device(values))
            return

        p.error(err_msg)


class Formatter(argparse.HelpFormatter):
    """
    Custom format for help message.
    """

    def _get_help_string(
        self, action: argparse.Action
    ) -> str | None:  # pragma: no cover
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


def parser(name: str = "dxtb", **kwargs: Any) -> argparse.ArgumentParser:
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

    p = argparse.ArgumentParser(
        description=desc,
        prog=name,
        formatter_class=lambda prog: Formatter(prog, max_help_position=60),
        add_help=False,
        **kwargs,
    )
    p.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="R|Show this help message and exit.",
    )
    p.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Show version and exit.",
    )
    p.add_argument(
        "-c",
        "--chrg",
        action=action_not_less_than(-10.0),
        type=int,
        default=defaults.CHRG,
        nargs="+",
        help="R|Molecular charge.",
    )
    p.add_argument(
        "--spin",
        "--uhf",
        action=action_not_less_than(0.0),
        type=int,
        default=defaults.SPIN,
        nargs="+",
        help="R|Molecular spin.",
    )
    p.add_argument(
        "--efield",
        type=float,
        nargs=3,
        help="R|Homogeneous electric field in V/Å.",
    )
    p.add_argument(
        "--exclude",
        type=str,
        default=defaults.EXCLUDE,
        choices=defaults.EXCLUDE_CHOICES,
        nargs="+",
        help="R|Turn off energy contributions.",
    )
    p.add_argument(
        "--etemp",
        action=action_not_less_than(0.0),
        type=float,
        default=defaults.FERMI_ETEMP,
        help="R|Electronic Temperature in K.",
    )
    p.add_argument(
        "--dtype",
        action=ConvertToTorchDtype,
        type=str,
        default=defaults.TORCH_DTYPE,
        choices=defaults.TORCH_DTYPE_CHOICES,
        help="R|Data type for PyTorch floating point tensors.",
    )
    p.add_argument(
        "--device",
        action=ConvertToTorchDevice,
        type=str,
        default=torch.device(defaults.TORCH_DEVICE),
        help="R|Device for PyTorch tensors.",
    )
    p.add_argument(
        "--fermi_maxiter",
        type=int,
        default=defaults.FERMI_MAXITER,
        help="R|Maximum number of iterations for Fermi smearing.",
    )
    p.add_argument(
        "--fermi_energy_partition",
        type=str,
        default=defaults.FERMI_PARTITION,
        choices=defaults.FERMI_PARTITION_CHOICES,
        help="R|Partitioning scheme for electronic free energy.",
    )
    p.add_argument(
        "--maxiter",
        type=int,
        default=defaults.MAXITER,
        help="R|Maximum number of SCF iterations.",
    )
    p.add_argument(
        "--mixer",
        type=str,
        default=defaults.MIXER,
        choices=defaults.MIXER_CHOICES,
        help="R|Mixing algorithm for convergence acceleration.",
    )
    p.add_argument(
        "--damp",
        type=float,
        default=defaults.DAMP,
        help="R|Damping factor for mixing in SCF iterations.",
    )
    p.add_argument(
        "--scf-mode",
        type=str,
        default=defaults.SCF_MODE,
        choices=defaults.SCF_MODE_CHOICES,
        help=(
            "R|Method of gradient tracking in SCF iterations.\n"
            " - 'default' is equivalent to 'implicit' and requests SCF\n"
            "   gradient computation via a single derivative facilitated\n"
            "   by the implicit function theorem.\n"
            " - 'full' and 'full_tracking' both request complete gradient\n"
            "   tracking through all SCF iterations.\n"
            " - 'experimental' converges the SCF without gradient tracking\n"
            "   then runs a single additional iteration to reconnect gradients."
        ),
    )
    p.add_argument(
        "--scp-mode",
        type=str,
        default=defaults.SCP_MODE,
        choices=defaults.SCP_MODE_CHOICES,
        help=(
            "R|Self-consistent parameter, i.e., which variable is converged\n"
            "in the SCF. Note that 'charge' and 'charges' is identical."
        ),
    )
    p.add_argument(
        "--int-driver",
        "--int_driver",
        type=str,
        default=defaults.INTDRIVER,
        choices=defaults.INTDRIVER_CHOICES,
        help=("R|Integral driver."),
    )

    p.add_argument(
        "--verbosity",
        type=int,
        default=defaults.VERBOSITY,
        help="R|Verbosity level of printout.",
    )
    p.add_argument(
        "-v",
        action="count",
        default=0,
        help="R|Increase verbosity level of printout by one.",
    )
    p.add_argument(
        "-s",
        action="count",
        default=0,
        help="R|Reduce verbosity level of printout by one.",
    )
    p.add_argument(
        "--loglevel",
        "--log-level",
        type=str,
        default=defaults.LOG_LEVEL,
        choices=defaults.LOG_LEVEL_CHOICES,
        help="R|Logging level.",
    )

    p.add_argument(
        "--method",
        type=str,
        default=defaults.METHOD,
        choices=defaults.METHOD_CHOICES,
        help="R|Method for calculation.",
    )
    p.add_argument(
        "--guess",
        type=str,
        default=defaults.GUESS,
        choices=defaults.GUESS_CHOICES,
        help="R|Model for initial charges.",
    )
    p.add_argument(
        "--grad",
        action="store_true",
        help="R|Whether to compute gradients for positions w.r.t. energy.",
    )
    p.add_argument(
        "--profile",
        action="store_true",
        help=(
            "R|Profile the program.\n"
            "Creates 'dxtb.profile' that can be analyzed with 'snakeviz'."
        ),
    )
    p.add_argument(
        "--xtol",
        type=float,
        default=defaults.X_ATOL,
        help="R|Set absolute tolerance for SCF (input).",
    )
    p.add_argument(
        "--ftol",
        type=float,
        default=defaults.F_ATOL,
        help="R|Set absolute tolerance for SCF (output).",
    )
    p.add_argument(
        "--detect-anomaly",
        action="store_true",
        help=("R|Enable PyTorch's anomaly detection mode."),
    )

    p.add_argument(
        "--dipole",
        "--dip",
        action="store_true",
        help=("R|Calculate the electric dipole moment."),
    )
    p.add_argument(
        "--dipole-numerical",
        "--dip-num",
        action="store_true",
        help=("R|Calculate the electric dipole moment numerically."),
    )
    p.add_argument(
        "--polarizability",
        "--pol",
        action="store_true",
        help=("R|Calculate the electric dipole polarizability."),
    )
    p.add_argument(
        "--polarizability-numerical",
        "--pol-num",
        action="store_true",
        help=("R|Calculate the electric dipole polarizability numerically."),
    )
    p.add_argument(
        "--hyperpolarizability",
        "--hyperpol",
        "--hpol",
        action="store_true",
        help=("R|Calculate the electric hyperpolarizability."),
    )
    p.add_argument(
        "--hyperpolarizability-numerical",
        "--hyperpol-num",
        "--hpol-num",
        action="store_true",
        help=("R|Calculate the electric hyperpolarizability numerically."),
    )

    p.add_argument(
        "--ir",
        action="store_true",
        help=("R|Calculate the IR spectrum."),
    )
    p.add_argument(
        "--ir-numerical",
        "--ir-num",
        action="store_true",
        help=("R|Calculate the IR spectrum numerically."),
    )
    p.add_argument(
        "--raman",
        action="store_true",
        help=("R|Calculate the Raman spectrum."),
    )
    p.add_argument(
        "--raman-numerical",
        "--raman-num",
        action="store_true",
        help=("R|Calculate the Raman spectrum numerically."),
    )

    p.add_argument(
        "--json",
        action="store_true",
    )

    p.add_argument(
        "--dir",
        nargs="*",
        type=is_dir,  # manual validation
        help="R|Directory with all files. Searches recursively.",
    )
    p.add_argument(
        "--filetype",
        type=str,
        choices=["xyz", "tm", "tmol", "turbomole", "json", "qcschema", "qm9"],
        help="R|Explicitly set file type of input.",
    )
    p.add_argument(
        "file",
        nargs="*",
        type=is_file,  # manual validation
        help="R|Path to coordinate file.",
    )

    return p
