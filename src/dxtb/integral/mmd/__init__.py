"""
McMurchie-Davidson algorithm
============================

This module contains two versions of the McMurchie-Davidson algorithm.

The differentiating factor is the calculation of the E-coefficients,
which are obtain from the well-known recursion relations or are explicitly
written down.
"""

from . import explicit, recursion

# set default
from .recursion import mmd_recursion as overlap_gto
from .recursion import mmd_recursion_gradient as overlap_gto_grad