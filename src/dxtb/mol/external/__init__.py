"""
External Representations
========================

Conversion between external molecule representations.
"""

try:
    from ._pyscf import *

    _has_pyscf = True
except ImportError as e:
    if "pyscf" in str(e).casefold():
        # If the error is specifically about the missing pyscf dependency,
        # we'll set `_has_pyscf` as False and leave an informative comment.
        _has_pyscf = False
    else:
        # If the error is about something else, we propagate it up.
        raise e


def is_pyscf_available() -> bool:
    """Check if PySCF is available."""
    return _has_pyscf
