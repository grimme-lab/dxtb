"""
McMurchie-Davidson algorithm
============================

This module contains two versions of the McMurchie-Davidson algorithm.

The differentiating factor is the calculation of the E-coefficients,
which are obtained from the well-known recursion relations or are explicitly
written down.

Note
----
The `recursion` module makes use of jit (tracing), which increases the start up
times of the program. Since the module is essentially never used (), we do not
explicitly import it here to avoid the jit start up.
"""
from . import explicit

# set default
from .explicit import md_explicit as overlap_gto
from .explicit import md_explicit_gradient as overlap_gto_grad
