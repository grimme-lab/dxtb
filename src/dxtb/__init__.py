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

# import timer first to get correct total time
from dxtb._src.timing import timer

timer.start("Import")
timer.start("PyTorch", parent_uid="Import")
import torch

timer.stop("PyTorch")
timer.start("dxtb", parent_uid="Import")

###############################################################################

from dxtb.__version__ import __version__


# order is important here
from dxtb._src.io import OutputHandler as OutputHandler
from dxtb._src.basis.indexhelper import IndexHelper as IndexHelper
from dxtb._src.calculators.base import Calculator
from dxtb._src.param import Param as Param
from dxtb._src.param.gfn1 import GFN1_XTB as GFN1_XTB
from dxtb._src.param.gfn2 import GFN2_XTB as GFN2_XTB


from dxtb import calculators as calculators
from dxtb import components as components
from dxtb import config as config
from dxtb import integrals as integrals
from dxtb import labels as labels
from dxtb import typing as typing

###############################################################################

# stop timers and remove from global namespace
del torch
timer.stop("dxtb")
timer.stop("Import")

###############################################################################

__all__ = [
    "calculators",
    "components",
    "Calculator",
    "GFN1_XTB",
    "GFN2_XTB",
    "IndexHelper",
    "timer",
    "__version__",
]
