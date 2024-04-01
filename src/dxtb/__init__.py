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
dxtb
====

A fully differentiable extended tight-binding package.
"""

from __future__ import annotations

# import timer first to get correct total time
from .timing import timer

timer.start("Import")
timer.start("PyTorch", parent_uid="Import")
import torch

timer.stop("PyTorch")
timer.start("dxtb", parent_uid="Import")

from dxtb.exlibs import scipy as scipy


from . import _types
from . import io

# import interaction before Coulomb to avoid circular import
from dxtb.components.interactions import Interaction, InteractionList
from dxtb.components.interactions.coulomb import (
    secondorder,
    thirdorder,
    ES2,
    ES3,
    new_es2,
    new_es3,
)

from dxtb.components import interactions as interactions
from dxtb.components import classicals as classicals

from dxtb.components.interactions import solvation as solvation
from dxtb.components.interactions import coulomb as coulomb
from dxtb.components.interactions import external as external

from dxtb.components.classicals import dispersion as dispersion
from dxtb.components.classicals import repulsion as repulsion
from dxtb.components.classicals import halogen as halogen

# from . import integral as ints
from .__version__ import __version__

from dxtb.basis import Basis, IndexHelper
from .components.classicals import Halogen, Repulsion, new_halogen, new_repulsion
from .components.classicals import DispersionD3, new_dispersion
from .mol import molecule
from .param import GFN1_XTB, Param
from .xtb import Calculator
from dxtb.utils import batch

timer.stop("dxtb")
timer.stop("Import")
