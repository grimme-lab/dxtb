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
Parametrizations: Overview
==========================

This module defines the parametrization of the extended tight-binding
Hamiltonians in the form of a hierarichical structure of Pydantic models.

The structure of the parametrization is adapted from the `tblite`_ library and
separates the species-specific parameter records from the general interactions
included in the method.

The standard parametrizations of GFN1 and GFN2 are predefined.

Parametrizations: Formats
=========================

Since `tblite`_ exports the parametrization in TOML format, we also read the
predefined parametrizations from the TOML files. However, the :class:`.Param`
class also supports reading JSON and YAML formats.
The parametrization can also be adapted and written back to the respective
formats.

.. code-block:: python

    from dxtb import Param, GFN1_XTB

    # Save existing GFN1-xTB parametrization to JSON file
    GFN1_XTB.to_file("gfn1-xtb.json")

    # Load the parametrization from the JSON file
    param = Param.from_file("gfn1-xtb.json")

    # compare the loaded parametrization with the original
    assert param == GFN1_XTB

.. _tblite: https://tblite.readthedocs.io
"""

from pydantic import __version__ as pydantic_version

if tuple(map(int, pydantic_version.split("."))) < (2, 0, 0):
    raise RuntimeError(
        "pydantic version outdated: dxtb requires pydantic >=2.0.0 "
        f"(version {pydantic_version} installed)."
    )


from .base import Param
from .gfn1 import GFN1_XTB
from .gfn2 import GFN2_XTB
from .utils import *

__all__ = ["Param", "GFN1_XTB", "GFN2_XTB"]
