"""
Main module and command line entrypoint for dxtb.
"""

import sys

from . import cli
from .__version import __version__

__all__ = ["__version__"]


def main() -> None:
    if getattr(sys, "frozen", False):
        # pylint: disable=import-outside-toplevel
        from multiprocessing import freeze_support

        freeze_support()

    if "--profile" in sys.argv:
        # pylint: disable=import-outside-toplevel
        import cProfile
        import pstats

        with cProfile.Profile() as profile:
            entry_point()

        stats = pstats.Stats(profile)
        stats.sort_stats(pstats.SortKey.TIME)

        # Use snakeviz to visualize the profile
        stats.dump_stats("dxtb.profile")
    else:
        entry_point()


def entry_point() -> None:
    """
    Wrapper for singlepoint driver.
    """

    args = cli.argparser().parse_args()

    if hasattr(args, "version") is True:
        print(f"dxtb {__version__}")
        sys.exit(0)

    if args.file is None:
        print("No coordinate file given.")
        sys.exit(0)

    d = cli.Driver(args)
    d.singlepoint()


if __name__ == "__main__":
    main()
