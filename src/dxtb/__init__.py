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

from dxtb.components import interactions

# from . import integral as ints
from .__version__ import __version__

from dxtb.basis import Basis, IndexHelper
from .components.classicals import Halogen, Repulsion, new_halogen, new_repulsion
from .components.classicals import DispersionD3, new_dispersion
from .components.interactions import external
from .mol import molecule
from .param import GFN1_XTB, Param
from .xtb import Calculator
from dxtb.components.interactions import solvation as solvation

timer.stop("dxtb")
timer.stop("Import")
