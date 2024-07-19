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
Integrals: Wrappers
===================

Shortcuts for integral calculations.
"""

from dxtb._src.integral.wrappers import dipint as dipint
from dxtb._src.integral.wrappers import hcore as hcore
from dxtb._src.integral.wrappers import overlap as overlap
from dxtb._src.integral.wrappers import quadint as quadint

__all__ = ["dipint", "hcore", "overlap", "quadint"]
