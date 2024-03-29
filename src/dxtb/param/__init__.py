"""
Extended tight-binding parametrization
======================================

This module defines the parametrization of the extended tight-binding Hamiltonian.

The structure of the parametrization is adapted from the `tblite`_ library and
separates the species-specific parameter records from the general interactions
included in the method.

.. _tblite: https://tblite.readthedocs.io
"""

from pydantic import __version__ as pydantic_version

if tuple(map(int, pydantic_version.split("."))) < (2, 0, 0):
    raise RuntimeError(
        "pydantic version outdated: dxtb requires pydantic >=2.0.0 "
        f"(version {pydantic_version} installed)."
    )


from .base import Param
from .charge import Charge
from .dispersion import Dispersion
from .element import Element
from .gfn1 import GFN1_XTB
from .gfn2 import GFN2_XTB
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import EffectiveRepulsion, Repulsion
from .thirdorder import ThirdOrder
from .util import *
