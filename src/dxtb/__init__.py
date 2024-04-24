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


from dxtb.typing import exceptions
from dxtb import io

from dxtb.exlibs import scipy as scipy

# import interaction bases before any implementation to avoid circular import
from dxtb.components.interactions import (
    Interaction as Interaction,
    InteractionList as InteractionList,
)

from dxtb.components import classicals as classicals
from dxtb.components import interactions as interactions

from dxtb.components.interactions import coulomb as coulomb
from dxtb.components.interactions.field import efield as efield
from dxtb.components.interactions import solvation as solvation

from dxtb.components.classicals import dispersion as dispersion
from dxtb.components.classicals import halogen as halogen
from dxtb.components.classicals import repulsion as repulsion

# classical components and their factories
from dxtb.components.classicals import (
    DispersionD3 as DispersionD3,
    DispersionD4 as DispersionD4,
    Halogen as Halogen,
    Repulsion as Repulsion,
    new_dispersion as new_dispersion,
    new_halogen as new_halogen,
    new_repulsion as new_repulsion,
)

# interaction components and their factories
from dxtb.components.interactions import (
    ElectricField as ElectricField,
    ElectricFieldGrad as ElectricFieldGrad,
    ES2 as ES2,
    ES3 as ES3,
    GeneralizedBorn as GeneralizedBorn,
    new_efield as new_efield,
    new_efield_grad as new_efield_grad,
    new_es2 as new_es2,
    new_es3 as new_es3,
    new_solvation as new_solvation,
)

# from . import integral as ints
from .__version__ import __version__

from dxtb.basis import Basis, IndexHelper
from .param import GFN1_XTB, GFN2_XTB, Param
from .xtb import Calculator
from dxtb.utils import batch

timer.stop("dxtb")
timer.stop("Import")

__all__ = [
    "Basis",
    "Calculator",
    "DispersionD3",
    "ES2",
    "ES3",
    "exceptions",
    "GFN1_XTB",
    "GFN2_XTB",
    "Halogen",
    "IndexHelper",
    "Interaction",
    "InteractionList",
    "Param",
    "Repulsion",
    "batch",
    "classicals",
    "coulomb",
    "dispersion",
    "efield",
    "interactions",
    "io",
    "new_dispersion",
    "new_es2",
    "new_es3",
    "new_halogen",
    "new_repulsion",
    "solvation",
    "scipy",
    "torch",
    "timer",
    "__version__",
]
