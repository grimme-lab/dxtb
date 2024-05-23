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
McMurchie-Davidson algorithm
============================

This module contains two versions of the McMurchie-Davidson algorithm.

The differentiating factor is the calculation of the E-coefficients,
which are obtained from the well-known recursion relations or are explicitly
written down.

Note
----
The `recursion` module makes use of jit (tracing), which increases the start up
times of the program. Since the module is essentially never used, we do not
explicitly import it here to avoid the jit start up.
"""

from . import explicit

# set default
from .explicit import md_explicit as overlap_gto
from .explicit import md_explicit_gradient as overlap_gto_grad
