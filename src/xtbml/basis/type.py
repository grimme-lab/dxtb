""" Definition of basic data classes """

import torch
from typing import List, Dict

from xtbml.param.element import Element
from xtbml.param.base import Param

from xtbml.basis.ortho import orthogonalize
from xtbml.basis.slater import slater_to_gauss


# Maximum contraction length of basis functions.
# The limit is chosen as twice the maximum size returned by the STO-NG expansion
maxg = 12

_aqm2lsh = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}


class Cgto_Type:
    """Contracted Gaussian type basis function"""

    def __init__(self):
        # Angular momentum of this basis function
        self.ang = -1
        # Contraction length of this basis function
        self.nprim = 0
        # Exponent of the primitive Gaussian functions
        self.alpha = torch.zeros(maxg)
        # Contraction coefficients of the primitive Gaussian functions,
        # might contain normalization
        self.coeff = torch.zeros(maxg)

    def __str__(self):
        return f"cgto( l:{self.ang} | ng:{self.nprim} | alpha:{self.alpha} | coeff:{self.coeff} )"

    def __repr__(self):
        return self.__str__()


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
