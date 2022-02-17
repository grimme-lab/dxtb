# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from typing import List, Dict, Tuple

from ..param import Param
from ..param.element import Element
from ..basis.ortho import orthogonalize
from ..basis.slater import slater_to_gauss
from ..basis.type import Cgto_Type

_aqm2lsh = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}

def _process_record(record: Element) -> List[Cgto_Type]:
    """
    Create basis set from element record
    """

    cgto = []

    nsh = len(record.shells)
    ang_idx = nsh * [-1]
    ortho = nsh * [None]

    lsh = [_aqm2lsh.get(shell[-1]) for shell in record.shells]
    pqn = [int(shell[0]) for shell in record.shells]

    for ish in range(nsh):
        il = lsh[ish]
        ng = record.ngauss[ish]

        if ang_idx[il] >= 0:
            ortho[ish] = ang_idx[il]
        else:
            ang_idx[il] = ish

        cgtoi = Cgto_Type()
        slater_to_gauss(ng, pqn[ish], il, record.slater[ish], cgtoi, True)
        cgto.append(cgtoi)

    for ish in range(nsh):
        if ortho[ish] is not None:
            orthogonalize(cgto[ortho[ish]], cgto[ish])

    return cgto


def _get_valence_shells(record: Element) -> List[bool]:

    valence = []

    nsh = len(record.shells)
    ang_idx = nsh * [-1]
    lsh = [_aqm2lsh.get(shell[-1]) for shell in record.shells]

    for ish in range(nsh):
        il = lsh[ish]

        valence.append(ang_idx[il] < 0)
        if valence[-1]:
            ang_idx[il] = ish

    return valence



class Basis:
    """
    Atomic orbital basis set definition.
    """

    cgto: Dict[str, List[Cgto_Type]] = {}
    """Definition of the atom centered basis functions for each species"""

    def __init__(self, species: List[str], par: Param) -> None:

        for isp in species:
            record = par.element[isp]
            self.cgto[isp] = _process_record(record)


class Hamiltonian:
    """
    Model to obtain the core Hamiltonian from the overlap matrix elements.
    """

    selfenergy: Dict[str, List[float]] = {}
    """Self-energy of each species"""

    kcn: Dict[str, List[float]] = {}
    """Coordination number dependent shift of the self energy"""

    shpoly: Dict[str, List[float]] = {}
    """Polynomial parameters for the distant dependent scaling"""

    refocc: Dict[str, List[float]] = {}
    """Reference occupation numbers"""

    hscale: Dict[Tuple[str, str], List[List[float]]] = {}
    """Off-site scaling factor for the Hamiltonian (not implemented)"""

    rad: Dict[str, float] = {}
    """Van-der-Waals radius of each species (not implemented)"""

    def __init__(self, species: List[str], par: Param):

        for isp in species:
            record = par.element[isp]

            _valence = _get_valence_shells(record)

            self.selfenergy[isp] = par.element[isp].levels.copy()
            self.kcn[isp] = par.element[isp].kcn.copy()
            self.shpoly[isp] = par.element[isp].shpoly.copy()

            self.refocc[isp] = [
                occ if val else 0.0
                for occ, val in zip(par.element[isp].refocc, _valence)
            ]


class Calculator:
    """
    Parametrized calculator defining the extended tight-binding model.

    The calculator holds the atomic orbital basis set for defining the Hamiltonian
    and the overlap matrix.
    """

    basis: Basis
    """Atomic orbital basis set definition."""

    hamiltonian: Hamiltonian
    """Core Hamiltonian definition."""

    def __init__(self, species: List[str], par: Param) -> None:

        self.basis = Basis(species, par)
        self.hamiltonian = Hamiltonian(species, par)
