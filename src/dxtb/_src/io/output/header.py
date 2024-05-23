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
Print a fancy header.
"""

from __future__ import annotations

__all__ = ["get_header"]


WIDTH = 70


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
    centered_lines = [line.center(WIDTH) for line in logo]
    # Join the lines with newlines and add the top border
    header = f"{WIDTH * '='}\n" + "\n".join(centered_lines) + f"\n{WIDTH * '='}\n"

    return header
