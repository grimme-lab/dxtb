"""
dxtb
====

A fully differentiable extended tight-binding package.
"""

from __future__ import annotations

import logging

from . import io
from .__version__ import __version__
from .basis import Basis, IndexHelper
from .bond import guess_bond_length, guess_bond_order
from .charges import ChargeModel, solve
from .classical import Halogen, Repulsion, new_halogen, new_repulsion
from .coulomb import ES2, ES3, new_es2, new_es3
from .dispersion import DispersionD3, new_dispersion
from .interaction import external
from .mol import molecule
from .param import GFN1_XTB, Param
from .solvation import GeneralizedBorn
from .xtb import Calculator

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s->%(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
