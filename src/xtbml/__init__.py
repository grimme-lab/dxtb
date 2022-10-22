"""
Main module and command line entrypoint for dxtb.
"""

import sys

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

    from .cli import Driver, argparser

    args = argparser().parse_args()

    if args.version:
        print(__version__)
        sys.exit(0)

    d = Driver(args)
    d.singlepoint()


if __name__ == "__main__":
    main()
