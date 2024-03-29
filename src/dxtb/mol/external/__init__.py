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
External Representations
========================

Conversion between external molecule representations.
"""

try:
    from ._pyscf import *

    _has_pyscf = True
except ImportError as e:
    if "pyscf" in str(e).casefold():
        # If the error is specifically about the missing pyscf dependency,
        # we'll set `_has_pyscf` as False and leave an informative comment.
        _has_pyscf = False
    else:
        # If the error is about something else, we propagate it up.
        raise e


def is_pyscf_available() -> bool:
    """Check if PySCF is available."""
    return _has_pyscf
