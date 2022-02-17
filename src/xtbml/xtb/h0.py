
import math
import torch

from ..integral.overlap import maxl, msao
from ..integral.multipole import multipole_cgto
from ..exlibs.tblite import shift_operator,dimDipole, dimQuadrupole

# TODO:
#   * tb_h0spec

class Tb_Hamiltonian():
    """ Implementation of the effective core Hamiltonian used in the extended tight binding. """

    def __init__(self):
        # Atomic level information
        self.selfenergy =  torch.zeroes([0,0])    
        # Coordination number dependence of the atomic levels
        self.kcn =  torch.zeroes([0,0])
        # Charge dependence of the atomic levels
        self.kq1 =  torch.zeroes([0,0])
        # Charge dependence of the atomic levels
        self.kq2 =  torch.zeroes([0,0])
        # Enhancement factor to scale the Hamiltonian elements
        self.hscale =  torch.zeroes([0,0,0,0])
        # Polynomial coefficients for distance dependent enhancement factor
        self.shpoly =  torch.zeroes([0,0])
        # Atomic radius for polynomial enhancement
        self.rad =  torch.zeroes([0])
        # Reference occupation numbers
        self.refocc =  torch.zeroes([0,0])
        return

    def __str__(self):
        # TODO:
        raise NotImplementedError
        return f"Tb_Hamiltonian( l:{self.ang} | ng:{self.nprim} | alpha:{self.alpha} | coeff:{self.coeff} )"


    def new_hamiltonian(self, mol, bas, spec):
        """ Constructor for a new Hamiltonian object, consumes a Hamiltonian specification

        Args:
            mol (structure_type): [description]
            bas (basis_type): [description]
            spec (tb_h0spec): [description]
        """ # TODO: docstring

        # TODO

        mshell = torch.max(bas.nsh_id)
        # allocate(self.selfenergy[mshell, mol.nid], self.kcn[mshell, mol.nid], &
        #     & self.kq1[mshell, mol.nid], self.kq2[mshell, mol.nid])


        # TODO: how to define interface? -- abstract base class needed?
        self.get_selfenergy(mol, bas, self.selfenergy)
        self.get_cnshift(mol, bas, self.kcn)
        self.get_q1shift(mol, bas, self.kq1)
        self.get_q2shift(mol, bas, self.kq2)

        # TODO: definde in abstract baseclass? -- no direct implementation in tblite/xtb
        # allocate(self.hscale[mshell, mshell, mol.nid, mol.nid])
        # self.get_hscale(mol, bas, self.hscale)
        # 
        # allocate(self.shpoly[mshell, mol.nid], self.rad[mol.nid])
        # self.get_rad[mol, bas, self.rad]
        # self.get_shpoly[mol, bas, self.shpoly]
        # 
        # allocate(self.refocc[mshell, mol.nid])
        # self.get_reference_occ(mol, bas, self.refocc)

        return self


    def get_selfenergy(h0, id, ish_at, nshell, selfenergy, cn=None, qat=None, dsedcn=None, dsedq=None):
        """[summary]

        Args:
            h0 ([type]): [description]
            id ([type]): [description]
            ish_at (bool): [description]
            nshell ([type]): [description]
            cn ([type]): [description]
            qat ([type]): [description]
            selfenergy ([type]): [description]
            dsedcn ([type]): [description]
            dsedq ([type]): [description]
        """ # TODO: docstring

        # type(tb_hamiltonian), intent(in) :: h0
        # integer, intent(in) :: id[:]
        # integer, intent(in) :: ish_at[:]
        # integer, intent(in) :: nshell[:]
        # real(wp), intent(in), optional :: cn[:]
        # real(wp), intent(in), optional :: qat[:]
        # real(wp), intent(out) :: selfenergy[:]
        # real(wp), intent(out), optional :: dsedcn[:]
        # real(wp), intent(out), optional :: dsedq[:]
        
        # TODO needs self?

        selfenergy = torch.zeros(selfenergy.size())
        if dsedcn is not None:
            dsedcn = torch.zeros(dsedcn.size())
        if dsedq is not None:
             dsedq = torch.zeros(dsedq.size())

        for iat in range(len(id)):
            izp = id[iat]
            ii = ish_at[iat]
            for ish in range(nshell[izp]):
                selfenergy[ii+ish] = h0.selfenergy[ish, izp]

        if cn is not None:
            if dsedcn is not None:
                for iat in range(len(id)):
                    izp = id[iat]
                    ii = ish_at[iat]
                    for ish in range(nshell[izp]):
                        selfenergy[ii+ish] = selfenergy[ii+ish] - h0.kcn[ish, izp] * cn[iat]
                        dsedcn[ii+ish] = -h0.kcn[ish, izp]
            else:
                for iat in range(len(id)):
                    izp = id[iat]
                    ii = ish_at[iat]
                    for ish in range(nshell[izp]):
                        selfenergy[ii+ish] = selfenergy[ii+ish] - h0.kcn[ish, izp] * cn[iat]

        if qat is not None:
            if dsedq is not None:
                for iat in range(len(id)):
                    izp = id[iat]
                    ii = ish_at[iat]
                    for ish in range(nshell[izp]):
                        selfenergy[ii+ish] = selfenergy[ii+ish] - h0.kq1[ish, izp]*qat[iat] - h0.kq2[ish, izp]*qat[iat]**2
                        dsedq[ii+ish] = -h0.kq1[ish, izp] - h0.kq2[ish, izp]*2*qat[iat]
            else:
                for iat in range(len(id)):
                    izp = id[iat]
                    ii = ish_at[iat]
                    for ish in range(nshell[izp]):
                        selfenergy[ii+ish] = selfenergy[ii+ish] - h0.kq1[ish, izp]*qat[iat] - h0.kq2[ish, izp]*qat[iat]**2
        
        return selfenergy, dsedcn, dsedq

    def get_hamiltonian(mol, trans, list, bas, h0, selfenergy, overlap, dpint, qpint, hamiltonian):

        # !> Molecular structure data
        # type(structure_type), intent(in) :: mol
        # !> Lattice points within a given realspace cutoff
        # real(wp), intent(in) :: trans(:, :)
        # !> Neighbour list
        # type(adjacency_list), intent(in) :: list
        # !> Basis set information
        # type(basis_type), intent(in) :: bas
        # !> Hamiltonian interaction data
        # type(tb_hamiltonian), intent(in) :: h0
        # !> Diagonal elememts of the Hamiltonian
        # real(wp), intent(in) :: selfenergy[:]
        # !> Overlap integral matrix
        # real(wp), intent(out) :: overlap(:, :)
        # !> Dipole moment integral matrix
        # real(wp), intent(out) :: dpint(:, :, :)
        # !> Quadrupole moment integral matrix
        # real(wp), intent(out) :: qpint(:, :, :)
        # !> Effective Hamiltonian
        # real(wp), intent(out) :: hamiltonian(:, :)

        '''integer :: iat, jat, izp, jzp, itr, k, img, inl
        integer :: ish, jsh, is, js, ii, jj, iao, jao, nao, ij
        real(wp) :: rr, r2, vec(3), cutoff2, hij, shpoly, dtmpj(3), qtmpj(6)
        real(wp), allocatable :: stmp[:], dtmpi(:, :), qtmpi(:, :)'''

        overlap = torch.zeros(overlap.size())
        dpint = torch.zeros(dpint.size())
        qpint = torch.zeros(qpint.size())
        hamiltonian = torch.zeros(hamiltonian.size())

        # TODO - allocate
        dtmpj = torch.zeros(dimDipole)
        qtmpj = torch.zeros(dimQuadrupole)
        # stmp = torch.zeros(msao(bas.maxl)**2)
        # dtmpi = torch.zeros(dimDipole, msao(bas.maxl)**2)
        # qtmpi = torch.zeros(dimQuadrupole, msao(bas.maxl)**2)

        for iat in range(mol.nat):
            izp = mol.id[iat]
            iss = bas.ish_at[iat]
            inl = list.inl[iat]
            for img in range(list.nnl[iat]):
                jat = list.nlat[img+inl]
                itr = list.nltr[img+inl]
                jzp = mol.id[jat]
                js = bas.ish_at[jat]
                vec = mol.xyz[:, iat] - mol.xyz[:, jat] - trans[:, itr] #TODO: check torch.tensor
                r2 = vec[1]**2 + vec[2]**2 + vec[3]**2
                rr = math.sqrt(math.sqrt(r2) / (h0.rad[jzp] + h0.rad[izp]))

                for ish in range(bas.nsh_id[izp]):
                    ii = bas.iao_sh[iss+ish]
                    for jsh in range(bas.nsh_id[jzp]):
                        jj = bas.iao_sh[js+jsh]

                        stmp, dtmpi, qtmpi = multipole_cgto(bas.cgto[jsh, jzp], bas.cgto[ish, izp], r2, vec, bas.intcut, stmp, dtmpi, qtmpi)

                        shpoly = (1.0 + h0.shpoly[ish, izp]*rr) * (1.0 + h0.shpoly[jsh, jzp]*rr)

                        hij = 0.5 * (selfenergy[iss+ish] + selfenergy[js+jsh]) * h0.hscale[jsh, ish, jzp, izp] * shpoly

                        nao = msao(bas.cgto[jsh, jzp].ang)

                        for iao in range(msao[bas.cgto[ish, izp].ang]):
                            for jao in range(nao):
                                ij = jao + nao*(iao-1)

                                # TODO
                                shift_operator(vec, stmp[ij], dtmpi[:, ij], qtmpi[:, ij], dtmpj, qtmpj)
  
                                overlap[jj+jao, ii+iao] += stmp[ij]

                                for k in range(dimDipole):                                    
                                    dpint[k, jj+jao, ii+iao] += dtmpi[k, ij]

                                for k in range(dimQuadrupole):
                                    qpint[k, jj+jao, ii+iao] +=  qtmpi[k, ij]
                                
                                hamiltonian[jj+jao, ii+iao] += stmp[ij] * hij

                                if iat != jat:
                                    overlap[jj+jao, ii+iao] += stmp[ij]    
                                    for k in range(dimDipole):      
                                        dpint[k, ii+iao, jj+jao] += qtmpj[k]                              
                                    for k in range(dimQuadrupole):
                                        qpint[k, ii+iao, jj+jao] +=  qtmpj[k]    
                                    hamiltonian[ii+iao, jj+jao] += stmp[ij] * hij
                      
        for iat in range(mol.nat):
            izp = mol.id[iat]
            iss = bas.ish_at[iat]
            vec[:] = 0.0
            r2 = 0.0
            rr = math.sqrt(math.sqrt(r2) / (h0.rad[izp] + h0.rad[izp]))

            for ish in range(bas.nsh_id[izp]):
                ii = bas.iao_sh(iss+ish)
                for jsh in range(bas.nsh_id[izp]):
                    jj = bas.iao_sh(iss+jsh)

                    # TODO
                    stmp, dtmpi, qtmpi = multipole_cgto(bas.cgto(jsh, izp), bas.cgto[ish, izp], r2, vec, bas.intcut, stmp, dtmpi, qtmpi)

                    shpoly = (1.0 + h0.shpoly[ish, izp]*rr) * (1.0 + h0.shpoly[jsh, izp]*rr)

                    hij = 0.5 * (selfenergy[iss+ish] + selfenergy[iss+jsh]) * shpoly

                    nao = msao(bas.cgto[jsh, izp].ang) #TODO: check slicable

                    for iao in range(msao[bas.cgto[ish, izp].ang]):
                        for jao in range(nao):
                            ij = jao + nao*(iao-1)
                            overlap[jj+jao, ii+iao] += stmp[ij]
                            dpint[:, jj+jao, ii+iao] += dtmpi[:, ij]
                            qpint[:, jj+jao, ii+iao] += qtmpi[:, ij]
                            hamiltonian[jj+jao, ii+iao] += stmp[ij] * hij

        return

    def  get_hamiltonian_gradient(mol, trans, list, bas, h0, selfenergy, dsedcn, pot, pmat, xmat, dEdcn, gradient, sigma):
        raise NotImplementedError

    def get_occupation(mol, bas, h0, nocc, n0at, n0sh):
        """[summary]

        Args:
            mol (structure_type): Molecular structure data
            bas (basis_type): Basis set information
            h0 (tb_hamiltonian): Hamiltonian interaction data
            nocc (float): Occupation number
            n0at (list[float]): Reference occupation for each atom
            n0sh (list[float]): Reference occupation for each shell
        """ # TODO

        nocc = -mol.charge
        n0at = torch.zeros(n0at.size())
        n0sh = torch.zeros(n0sh.size())

        for iat in range(mol.nat):
            izp = mol.id[iat]
            ii = bas.ish_at[iat]
            for ish in range(bas.nsh_id[izp]): 
                nocc = nocc + h0.refocc[ish, izp]
                n0at[iat] += h0.refocc[ish, izp]
                n0sh[ii+ish] += h0.refocc[ish, izp]

        return nocc, n0at, n0sh

