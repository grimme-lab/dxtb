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
Self-consistent field (SCF)
===========================

Definition of the self-consistent iterations.
"""

from .base import *
from .guess import get_guess
from .iterator import SelfConsistentField, solve
from .scf_full import BaseTSCF
from .scf_implicit import BaseXSCF
from .utils import get_density as get_density
