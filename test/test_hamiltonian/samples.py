"""
Molecules for testing the Hamiltonian. Reference values are stored in npz file.
"""

from ..molecules import mols

from xtbml.typing import Molecule

extra: dict[str, Molecule] = {
    "H2_nocn": {
        "numbers": mols["H2"]["numbers"],
        "positions": mols["H2"]["positions"],
    },
    "SiH4_nocn": {
        "numbers": mols["SiH4"]["numbers"],
        "positions": mols["SiH4"]["positions"],
    },
}


samples: dict[str, Molecule] = {**mols, **extra}
