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
Libcint Interface
=================

All the functions and classes required to interface with the `libcint` library.
"""

try:
    from tad_libcint.basis import AtomCGTOBasis, CGTOBasis
    from tad_libcint.interface.integrals import int1e, overlap
    from tad_libcint.interface.wrapper import LibcintWrapper
except ImportError as e:
    raise ImportError(
        f"Failed to import required modules. {e}. {e.name} provides a Python "
        "interface to the 'libcint' library for fast integral evaluation. "
        "It can be installed via 'pip install tad-libcint'."
    ) from e

__all__ = ["int1e", "overlap", "LibcintWrapper", "AtomCGTOBasis", "CGTOBasis"]
