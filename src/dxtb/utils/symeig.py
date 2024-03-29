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
A collection of common mathematical functions.

This module contains a collection of batch-operable, back-propagatable
mathematical functions.

Taken from TBMaLT.
https://github.com/tbmalt/tbmalt/blob/main/tbmalt/common/maths/__init__.py
"""

from __future__ import annotations

from tad_mctc.storch.linalg import eighb

__all__ = ["eighb"]
