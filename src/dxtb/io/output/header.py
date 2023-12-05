"""
Print a fancy header.
"""
from __future__ import annotations

width = 70


def get_header() -> str:
    logo = [
        r"      _      _   _      ",
        r"     | |    | | | |     ",
        r"   __| |_  _| |_| |__   ",
        r"  / _` \ \/ / __| '_ \  ",
        r" | (_| |>  <| |_| |_) | ",
        r"  \__,_/_/\_\\__|_.__/  ",
        r"                        ",
    ]

    # Center each line within the given width
    centered_lines = [line.center(width) for line in logo]
    # Join the lines with newlines and add the top border
    header = f"{width * '='}\n" + "\n".join(centered_lines) + f"\n{width * '='}\n"

    return header
