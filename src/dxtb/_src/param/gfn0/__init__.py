# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
Parametrization: GFN0-xTB
=========================

This module provides the GFN0-xTB parametrization.
The parametrization is stored in a TOML file and loaded lazily.

Example
-------

.. code-block:: python

    from dxtb._src.param.gfn2 import GFN0_XTB
    #from dxtb import GFN0_XTB  # also available from the top-level package

    # Initially, the parameters are not loaded
    print(GFN0_XTB._loaded is None)  # Expected output: True

    # Access the metadata to trigger loading
    m = GFN0_XTB.meta

    # Now, the parameters should be loaded
    print(GFN0_XTB._loaded is None)  # Expected output: False
"""

from .load import GFN0_XTB as GFN0_XTB
