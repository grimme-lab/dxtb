from math import sqrt
import torch
from typing import List, Dict, Tuple, Union, Optional

from ..adjlist import AdjacencyList
from ..basis import IndexHelper
from ..basis.type import Basis
from ..constants import EV2AU
from ..constants import FLOAT64 as DTYPE
from ..data.atomicrad import get_atomic_rad
from ..exlibs.tbmalt import Geometry
from ..integral import mmd
from ..param import Param, Element
from ..typing import Tensor


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

    mol: Geometry
    """Geometry representation"""

    par: Param
    """Representation of parametrization of xtb model"""

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
            kii = shell[2 * _lsh2aqm[ish]]
            for jsh in range(lmax):
                kjj = shell[2 * _lsh2aqm[jsh]]
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

                self.hscale[(isp, jsp)] = get_hscale(
                    lsh[isp], lsh[jsp], ri, rj, valence[isp], valence[jsp], km, ksh
                )

    def get_selfenergy(
        self,
        basis: Basis,
        cn: Optional[Tensor] = None,
        qat=None,
        dsedcn=None,
        dsedq=None,
    ):

        # calculate selfenergy using hamiltonian.selfenergy dict
        self_energy = torch.zeros(basis.nsh_tot, dtype=DTYPE)
        for i, sym in enumerate(self.mol.chemical_symbols):
            ii = int(basis.ish_at[i].item())
            for ish in range(basis.shells[sym]):
                self_energy[ii + ish] = self.selfenergy[sym][ish]

        if dsedcn is not None:
            dsedcn = torch.zeros(basis.nsh_tot, dtype=DTYPE)
        if dsedq is not None:
            dsedq = torch.zeros(basis.nsh_tot, dtype=DTYPE)

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

    def get_occupation(self, ihelp: IndexHelper):
        """
        Obtain the reference occupation numbers for each orbital.
        """

        occupation = torch.zeros(*ihelp.orbitals_to_shell.shape, dtype=DTYPE)

        for idx, sym in enumerate(self.mol.chemical_symbols):
            shell_index = ihelp.shell_index[idx]

            for ish in range(ihelp.shells_per_atom[idx]):
                orbital_index = ihelp.orbital_index[shell_index + ish]
                orbitals_per_shell = ihelp.orbitals_per_shell[shell_index + ish]

                for iao in range(orbitals_per_shell):
                    occupation[orbital_index + iao] = (
                        self.refocc[sym][ish] / orbitals_per_shell
                    )

        return occupation

    def build(
        self, basis: Basis, adjlist: AdjacencyList, cn: Optional[Tensor]
    ) -> Tuple[Tensor, Tensor]:
        self_energy = self.get_selfenergy(basis, cn)

        # init matrices
        h0 = torch.zeros(basis.nao_tot, basis.nao_tot, dtype=DTYPE)
        overlap = torch.zeros(basis.nao_tot, basis.nao_tot, dtype=DTYPE)

        # fill diagonal
        torch.diagonal(overlap, dim1=-2, dim2=-1).fill_(1.0)
        h0 = self.build_diagonal_blocks(h0, basis, self_energy)

        # fill off-diagonal
        h0, overlap = self.build_diatomic_blocks(
            self.mol, h0, overlap, basis, adjlist, self_energy
        )

        return h0, overlap

    def build_diagonal_blocks(
        self, h0: Tensor, basis: Basis, self_energy: Tensor
    ) -> Tensor:
        for i, element in enumerate(self.mol.chemical_symbols):
            iss = basis.ish_at[i].item()
            for ish in range(basis.shells[element]):
                ii = basis.iao_sh[iss + ish].item()
                hii = self_energy[iss + ish]

                i_nao = 2*basis.cgto[element][ish].ang + 1
                for iao in range(i_nao):
                    h0[ii + iao, ii + iao] = hii

        return h0

    def build_diatomic_blocks(
        self,
        mol: Geometry,
        h0: Tensor,
        overlap: Tensor,
        basis: Basis,
        adjlist: AdjacencyList,
        self_energy: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Construction of off-diagonal blocks of the Hamiltonian and overlap integrals.
        """

        for i, el_i in enumerate(mol.chemical_symbols):
            isa = basis.ish_at[i].item()
            inl = adjlist.inl[i].item()

            imgs = int(adjlist.nnl[i].item())
            for img in range(imgs):
                j = adjlist.nlat[img + inl].item()
                jsa = basis.ish_at[j].item()
                el_j = mol.chemical_symbols[j]

                vec = mol.positions[i, :] - mol.positions[j, :]
                r2 = torch.sum(vec**2)
                rr = torch.sqrt(torch.sqrt(r2) / (self.rad[el_i] + self.rad[el_j]))

                for ish in range(basis.shells[el_i]):
                    ii = basis.iao_sh[isa + ish].item()
                    for jsh in range(basis.shells[el_j]):
                        jj = basis.iao_sh[jsa + jsh].item()

                        cgtoi = basis.cgto[el_i][ish]
                        cgtoj = basis.cgto[el_j][jsh]

                        stmp = mmd.overlap(
                            (cgtoi.ang, cgtoj.ang),
                            (cgtoi.alpha[:cgtoi.nprim], cgtoj.alpha[:cgtoj.nprim]),
                            (cgtoi.coeff[:cgtoi.nprim], cgtoj.coeff[:cgtoj.nprim]),
                            -vec,
                        )

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

                        i_nao = 2*cgtoi.ang + 1
                        j_nao = 2*cgtoj.ang + 1
                        for iao in range(i_nao):
                            for jao in range(j_nao):
                                ij = jao + j_nao * iao

                                overlap[jj + jao, ii + iao].add_(stmp[iao, jao])
                                h0[jj + jao, ii + iao].add_(stmp[iao, jao] * hij)

                                if i != j:
                                    overlap[ii + iao, jj + jao].add_(stmp[iao, jao])
                                    h0[ii + iao, jj + jao].add_(stmp[iao, jao] * hij)

        return h0, overlap


def _get_valence_shells(record: Element) -> List[bool]:

    valence = []

    nsh = len(record.shells)
    ang_idx = nsh * [-1]
    lsh = [_aqm2lsh[shell[-1]] for shell in record.shells]

    for ish in range(nsh):
        il = lsh[ish]

        valence.append(ang_idx[il] < 0)
        if valence[-1]:
            ang_idx[il] = ish

    return valence
