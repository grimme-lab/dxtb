"""
Libcint Integrals
=================

This module contains the interface for integral calculation using the libcint
library. Derivatives are implemented analytically while retaining a fully
functional backpropagation.

This subpackage was heavily inspired by `DQC <https://github.com/diffqc/dqc>`__.
"""

from .intor import *
from .namemanager import *
from .symmetry import *
from .wrapper import *
