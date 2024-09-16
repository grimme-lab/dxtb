"""
Exlibs: Check Availablity
=========================

Simple check for the availability of external libraries.
"""

try:
    from tad_libcint import __version__  # type: ignore

    has_libcint = True
except ImportError:
    has_libcint = False

try:
    from pyscf import __version__  # type: ignore

    has_pyscf = True
except ImportError:
    has_pyscf = False

try:
    from scipy import __version__  # type: ignore

    has_scipy = True
except ImportError:
    has_scipy = False


__all__ = ["has_libcint", "has_pyscf", "has_scipy"]
