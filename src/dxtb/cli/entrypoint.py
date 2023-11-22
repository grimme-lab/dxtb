"""
Entrypoint for command line interface.
"""

from __future__ import annotations

import logging
import sys

from .. import __version__
from .._types import Sequence
from .argparser import parser
from .driver import Driver

logger = logging.getLogger(__name__)


def console_entry_point(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover
    """
    Entry point for CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command line arguments. Defaults to `None`.
        Only passed in tests.

    Returns
    -------
    int
        Exit status (from `entry_point_wrapper`).
    """
    if getattr(sys, "frozen", False):
        # pylint: disable=import-outside-toplevel
        from multiprocessing import freeze_support

        freeze_support()

    if "--profile" in sys.argv:
        # pylint: disable=import-outside-toplevel
        import cProfile
        import pstats

        with cProfile.Profile() as profile:
            ret = entry_point_wrapper(argv)

        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)

        # Use snakeviz to visualize the profile
        stats.dump_stats("dxtb.profile")

        return ret

    return entry_point_wrapper(argv)


def entry_point_wrapper(argv: Sequence[str] | None = None) -> int:
    """
    Wrapper for singlepoint driver.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command line arguments. Defaults to `None`.
        Only passed in tests.

    Returns
    -------
    int
        Exit status.

    Raises
    ------
    SystemExit
        Exits if `--version` flag found or no file given.
    """

    args = parser().parse_args(argv)

    if args.file is None or len(args.file) == 0:
        logger.info("No coordinate file given.")
        raise SystemExit(1)

    d = Driver(args)
    d.singlepoint()

    return 0
