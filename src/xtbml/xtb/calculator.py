# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from math import sqrt
from typing import List, Dict, Tuple, Union
import torch

from ..param import Param
from ..param.element import Element
from ..basis.ortho import orthogonalize
from ..basis.slater import slater_to_gauss
from ..basis.type import Cgto_Type
from ..data.atomicrad import get_atomic_rad

_aqm2lsh = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
}
_lsh2aqm = {
    0: "s",
    1: "p",
    2: "d",
    3: "f",
    4: "g",
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

    hscale: Dict[Tuple[str, str], Union[List[float], any]] = {}
    """Off-site scaling factor for the Hamiltonian"""

    rad: Dict[str, float] = {}
    """Van-der-Waals radius of each species"""

    def __init__(self, species: List[str], par: Param):

        lmax = 0
        lsh = {}
        valence = {}
        for isp in species:
            record = par.element[isp]

            lsh[isp] = [_aqm2lsh.get(shell[-1]) for shell in record.shells]
            valence[isp] = _get_valence_shells(record)
            lmax = max(lmax, *lsh[isp])

            self.selfenergy[isp] = par.element[isp].levels.copy()
            self.kcn[isp] = par.element[isp].kcn.copy()
            self.shpoly[isp] = par.element[isp].shpoly.copy()

            self.refocc[isp] = [
                occ if val else 0.0
                for occ, val in zip(par.element[isp].refocc, valence[isp])
            ]

            self.rad[isp] = get_atomic_rad(isp)
        lmax += 1

        # Collect shell specific scaling block
        #
        # FIXME: tblite implicitly spreads missing angular momenta, however this
        #        is only relevant for f shells and higher in present parametrizations.
        shell = par.hamiltonian.xtb.shell
        ksh = torch.zeros((lmax, lmax))
        for ish in range(lmax):
            kii = shell.get(2 * _lsh2aqm[ish])
            for jsh in range(lmax):
                kjj = shell.get(2 * _lsh2aqm[jsh])
                kij = (
                    _lsh2aqm[ish] + _lsh2aqm[jsh]
                    if jsh > ish
                    else _lsh2aqm[jsh] + _lsh2aqm[ish]
                )
                ksh[ish, jsh] = shell.get(kij, (kii + kjj) / 2)

        def get_hscale(li, lj, ri, rj, vi, vj, km, ksh):
            """Calculate Hamiltonian scaling for a shell block"""
            ni, nj = len(li), len(lj)
            hscale = torch.zeros((ni, nj))
            for ish in range(ni):
                kii = ksh[li[ish], li[ish]] if vi[ish] else par.hamiltonian.xtb.kpol
                for jsh in range(nj):
                    kjj = ksh[lj[jsh], lj[jsh]] if vj[jsh] else par.hamiltonian.xtb.kpol
                    zi = ri.slater[ish]
                    zj = rj.slater[ish]
                    zij = (2 * sqrt(zi * zj) / (zi + zj)) ** par.hamiltonian.xtb.wexp
                    hscale[ish, jsh] = zij * (
                        km * ksh[li[ish], lj[jsh]]
                        if vi[ish] and vj[jsh]
                        else (kii + kjj) / 2
                    )

            return hscale

        kpair = par.hamiltonian.xtb.kpair
        for isp in species:
            ri = par.element[isp]
            for jsp in species:
                rj = par.element[jsp]
                enp = 1.0 + par.hamiltonian.xtb.enscale * (ri.en - rj.en) ** 2

                km = kpair.get(f"{isp}-{jsp}", kpair.get(f"{jsp}-{isp}", 1.0)) * enp
                self.hscale[(isp, jsp)] = get_hscale(
                    lsh[isp], lsh[jsp], ri, rj, valence[isp], valence[jsp], km, ksh
                )


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
