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
Molecules for testing the Hamiltonian. Reference values are stored in npz file.
"""

from __future__ import annotations

from tad_mctc.data.molecules import mols

from dxtb._src.typing import Molecule

extra: dict[str, Molecule] = {
    "H2_nocn": {
        "numbers": mols["H2"]["numbers"],
        "positions": mols["H2"]["positions"],
    },
    "SiH4_nocn": {
        "numbers": mols["SiH4"]["numbers"],
        "positions": mols["SiH4"]["positions"],
    },
}


samples: dict[str, Molecule] = {**mols, **extra}
