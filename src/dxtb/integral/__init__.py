"""
Functions for calculation of overlap with McMurchie-Davidson algorithm.
"""

from .mmd.explicit import mmd_explicit
from .mmd.recursion import mmd_recursion
from .overlap import Overlap

# default to explicit
from .mmd import overlap_gto
