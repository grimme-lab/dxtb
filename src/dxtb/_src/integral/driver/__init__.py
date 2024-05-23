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
Integrals: Drivers
==================

Integral drivers are the main interface to the integral implementations. They
provide a unified interface to the integral implementations, and are responsible
for the calculation of the integrals.

There are two main types of integral drivers: `PyTorch` and `Libcint`. Note that
the `Libcint` drivers are only available if the
`tad-libcint <https://github.com/tad-mctc/tad-libcint>`__ library is installed.
The `PyTorch` drivers are implemented in pure Python, but are currently only
available for overlap integrals.
"""
# no imports here to allow lazy loading of drivers
