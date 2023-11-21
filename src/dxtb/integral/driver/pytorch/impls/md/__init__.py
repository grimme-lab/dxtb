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
from .explicit import md_explicit as overlap_gto
from .explicit import md_explicit_gradient as overlap_gto_grad
