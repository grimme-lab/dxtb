"""
Version module for dxtb.
"""

from __future__ import annotations

import torch


def version_tuple(version_string: str) -> tuple[int, ...]:
    """
    Convert a version string (with possible additional version specifications)
    to a tuple following semantic versioning.

    Parameters
    ----------
    version_string : str
        Version string to convert.

    Returns
    -------
    tuple[int, ...]
        Semantic version number as tuple.
    """
    main_version_part = version_string.split("-")[0].split("+")[0].split("_")[0]

    s = main_version_part.split(".")
    if 3 > len(s):
        raise RuntimeError(
            "Version specification does not seem to follow the semantic "
            f"versioning scheme of MAJOR.MINOR.PATCH ({s})."
        )

    version_numbers = [int(part) for part in s[:3]]
    return tuple(version_numbers)


__version__ = "0.0.1"
__tversion__ = version_tuple(torch.__version__)
