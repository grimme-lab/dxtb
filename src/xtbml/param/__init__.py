# This file is part of xtbml.

"""
This module defines the parametrization of the extended tight-binding Hamiltonian.

The structure of the parametrization is adapted from the `tblite`_ library and
separates the species-specific parameter records from the general interactions
included in the method.

.. _tblite: https://tblite.readthedocs.io
"""

from .base import Param
from .charge import Charge
from .dispersion import Dispersion
from .element import Element
from .gfn1 import GFN1_XTB
from .halogen import Halogen
from .hamiltonian import Hamiltonian
from .meta import Meta
from .repulsion import EffectiveRepulsion, Repulsion
from .util import (
    get_elem_param,
    get_elem_param_dict,
    get_elem_param_shells,
    get_pair_param,
)
from .thirdorder import ThirdOrder
