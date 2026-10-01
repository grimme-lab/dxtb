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
Integral implementations
========================

- ``pairs``: the pair builder, which enumerates the shell pairs of a molecule
  and assembles the overlap, its gradient, and the dipole and quadrupole
  integrals.
- ``pipeline``: the 3D assembly of one class of shell pairs from a 1D kernel.
- ``kernels``: the interchangeable 1D kernels, Obara-Saika (``os``, default)
  and McMurchie-Davidson with Hermite moments (``md``), selected by
  ``int_algorithm``.
- ``legacy``: the original loop-based overlap with explicit
  McMurchie-Davidson E-coefficients, used only by the legacy driver.
- ``trafo``: Cartesian-to-spherical transformation, shared by all of them.
"""
