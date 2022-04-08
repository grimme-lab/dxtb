# This file is part of xtbml.

"""
Base calculator for the extended tight-binding model.
"""

from math import sqrt
import torch
from typing import List, Dict, Tuple, Union

from xtbml.adjlist import AdjacencyList
from xtbml.basis.type import Basis
from xtbml.constants import EV2AU
from xtbml.data.atomicrad import get_atomic_rad
from xtbml.exlibs.tbmalt import Geometry
from xtbml.integral.overlap import msao, overlap_cgto
from xtbml.param import Param, Element


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

    def __init__(self, mol: Geometry, par: Param):
        self.mol = mol
        self.par = par

        lmax = 0
        lsh = {}
        valence = {}
        species = mol.chemical_symbols
        for isp in species:
            record = par.element[isp]

            lsh[isp] = [_aqm2lsh.get(shell[-1]) for shell in record.shells]
            valence[isp] = _get_valence_shells(record)
            lmax = max(lmax, *lsh[isp])

            self.selfenergy[isp] = [i * EV2AU for i in par.element[isp].levels]
            self.kcn[isp] = [i * EV2AU for i in par.element[isp].kcn]
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
                    zj = rj.slater[jsh]
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

                # print("\nisp", isp, "jsp", jsp, "lsh[isp]", lsh[isp], "lsh[jsp]", lsh[jsp])
                # print("ri", ri)
                # print("rj", rj)
                # print("")
                # print("valence[isp]", valence[isp], "valence[jsp]", valence[jsp])
                # print("km", km, "ksh", ksh)
                self.hscale[(isp, jsp)] = get_hscale(
                    lsh[isp], lsh[jsp], ri, rj, valence[isp], valence[jsp], km, ksh
                )

    def get_selfenergy(
        self,
        basis: Basis,
        cn: Union[torch.Tensor, None] = None,
        qat=None,
        dsedcn=None,
        dsedq=None,
    ):

        # calculate selfenergy using hamiltonian.selfenergy dict
        self_energy = torch.zeros(basis.nsh_tot)
        for i, sym in enumerate(self.mol.chemical_symbols):
            ii = int(basis.ish_at[i].item())
            for ish in range(basis.shells[sym]):
                self_energy[ii + ish] = self.selfenergy[sym][ish]

        if dsedcn is not None:
            dsedcn = torch.zeros(basis.nsh_tot)
        if dsedq is not None:
            dsedq = torch.zeros(basis.nsh_tot)

        if cn is not None:
            if dsedcn is not None:
                for i, sym in enumerate(self.mol.chemical_symbols):
                    ii = int(basis.ish_at[i].item())
                    for ish in range(basis.shells[sym]):
                        self_energy[ii + ish] -= self.kcn[sym][ish] * cn[i]
                        dsedcn[ii + ish] = -self.kcn[sym][ish]
            else:
                for i, sym in enumerate(self.mol.chemical_symbols):
                    ii = int(basis.ish_at[i].item())
                    for ish in range(basis.shells[sym]):
                        self_energy[ii + ish] -= self.kcn[sym][ish] * cn[i]

        # TODO:
        #  - requires init of self.kq1 and self.kq2
        #  - figuring out return values
        #
        # if qat is not None:
        #     if dsedq is not None:
        #         for i, sym in enumerate(self.mol.chemical_symbols):
        #             ii = int(basis.ish_at[i].item())
        #             for ish in range(basis.shells[sym]):
        #                 self_energy[ii + ish] -= (
        #                     self.kq1[sym][ish] * qat[i] - self.kq2[sym][ish] * qat[i]**2
        #                 )
        #                 dsedq[ii + ish] = (
        #                     -self.kq1[sym][ish] - self.kq2[sym][ish] * 2 * qat[i]
        #                 )

        #     else:
        #         for i, sym in enumerate(self.mol.chemical_symbols):
        #             ii = int(basis.ish_at[i].item())
        #             for ish in range(basis.shells[sym]):
        #                 self_energy[ii + ish] -= (
        #                     self.kq1[sym][ish] * qat[i] - self.kq2[sym][ish] * qat[i]
        #                 )

        return self_energy  # , dsedcn, dsedq

    def build(
        self, basis: Basis, adjlist: AdjacencyList, cn: Union[torch.Tensor, None]
    ):
        self_energy = self.get_selfenergy(basis, cn)

        # init matrices
        h0 = torch.zeros(basis.nao_tot, basis.nao_tot)
        overlap_int = torch.zeros(basis.nao_tot, basis.nao_tot)

        # fill diagonal
        h0 = self.buildDiagonalBlocks(h0, basis, self_energy)

        # fill off-diagonal
        h0, overlap_int = self.buildDiatomicBlocks(
            h0, overlap_int, basis, adjlist, self_energy
        )

        return h0, overlap_int

    def buildDiagonalBlocks(self, h0, basis, self_energy):
        for i, element in enumerate(self.mol.chemical_symbols):
            iss = basis.ish_at[i].item()
            for ish in range(basis.shells[element]):
                ii = basis.iao_sh[iss + ish].item()
                hii = self_energy[iss + ish]

                # print("izp", element, "iss", iss, "ish", ish, "ii", ii)

                i_nao = msao[basis.cgto[element][ish].ang]
                for iao in range(i_nao):
                    h0[ii + iao, ii + iao] = hii

        return h0

    def buildDiatomicBlocks(self, h0, overlap_int, basis, adjlist, self_energy):
        for i, el_i in enumerate(self.mol.chemical_symbols):
            isa = basis.ish_at[i].item()
            inl = adjlist.inl[i].item()

            imgs = int(adjlist.nnl[i].item())
            for img in range(imgs):
                j = adjlist.nlat[img + inl].item()
                itr = adjlist.nltr[img + inl].item()
                jsa = basis.ish_at[j].item()
                el_j = self.mol.chemical_symbols[j]

                vec = torch.sub(self.mol.positions[i, :], self.mol.positions[j, :])
                vec = torch.sub(vec, adjlist.trans[itr, :])
                r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
                rr = sqrt(sqrt(r2) / (self.rad[el_i] + self.rad[el_j]))

                for ish in range(basis.shells[el_i]):
                    ii = basis.iao_sh[isa + ish].item()
                    for jsh in range(basis.shells[el_j]):
                        jj = basis.iao_sh[jsa + jsh].item()

                        cgtoi = basis.cgto[el_i][ish]
                        cgtoj = basis.cgto[el_j][jsh]

                        stmp = overlap_cgto(cgtoj, cgtoi, r2, vec, basis.intcut)
                        stmp = torch.flatten(stmp)
                        # print("el_i", el_i, "el_j", el_j, "overlap", overlap)

                        shpoly = (1.0 + self.shpoly[el_i][ish] * rr) * (
                            1.0 + self.shpoly[el_j][jsh] * rr
                        )

                        hscale = self.hscale[(el_i, el_j)][ish, jsh].item()
                        hij = (
                            0.5
                            * (self_energy[isa + ish] + self_energy[jsa + jsh])
                            * hscale
                            * shpoly
                        )

                        # print("izp", element, "iss", iss,
                        #   "ish", ish, "jsh", jsh, "ii", ii, "jj", jj)
                        # print("hij", hij)

                        i_nao = msao[cgtoi.ang]
                        j_nao = msao[cgtoj.ang]
                        for iao in range(i_nao):
                            for jao in range(j_nao):
                                ij = jao + j_nao * iao

                                overlap_int[jj + jao, ii + iao] += stmp[ij] * hij
                                h0[jj + jao, ii + iao] += stmp[ij] * hij

                                if i != j:
                                    overlap_int[ii + iao, jj + jao] += stmp[ij]
                                    h0[ii + iao, jj + jao] += stmp[ij] * hij

        return h0, overlap_int


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

    def __init__(self, mol: Geometry, par: Param, acc: float = 1.0) -> None:

        self.basis = Basis(mol, par, acc)
        self.hamiltonian = Hamiltonian(mol, par)
