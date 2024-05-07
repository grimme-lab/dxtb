# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Entrypoint for command line interface.
"""

from __future__ import annotations

import logging
import sys

from dxtb import __version__
from dxtb._src.typing import Sequence

from .argparser import parser
from .driver import Driver

__all__ = ["console_entry_point", "entry_point_wrapper"]


logger = logging.getLogger(__name__)


def console_entry_point(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover
    """
    Entry point for CLI.

    Parameters
    ----------
    argv : Sequence[str] | None, optional
        Command line arguments. Defaults to ``None``.
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
        Command line arguments. Defaults to ``None``.
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
