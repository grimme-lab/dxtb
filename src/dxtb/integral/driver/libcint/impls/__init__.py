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
Libcint Integrals
=================

This module contains the interface for integral calculation using the libcint
library. Derivatives are implemented analytically while retaining a fully
functional backpropagation.

This subpackage was heavily inspired by `DQC <https://github.com/diffqc/dqc>`__.
"""

from .intor import *
from .namemanager import *
from .wrapper import *
