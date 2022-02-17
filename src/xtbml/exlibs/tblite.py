
# TODO: 
#   * Docstring
#   * h0 hamiltonian type object

import torch
import math

from ..integral.multipole import multipole_cgto
from ..integral.overlap import maxl, msao

# Number of dipole components used in tblite library (x, y, z)
dimDipole = 3
# Number of quadrupole components used in tblite library (xx, xy, yy, xz, yz, zz)
dimQuadrupole = 6

def shift_operator(vec, s, di, qi, dj, qj):
    """ 
    Shift multipole operator from Ket function (center i) to Bra function (center j),
    the multipole operator on the Bra function can be assembled from the lower moments
    on the Ket function and the displacement vector using horizontal shift rules.

    This is usually done inside the tblite library, but since we want to have both
    Bra and Ket contributions at once and do not want to iterate over both triangles
    of the multipole integral matrix we perform the shift of the moment operator here.

    Args:
        vec (list[float]): Displacement vector of center i and j
        s (float): Overlap integral between basis functions
        di (list[float]): Dipole integral with operator on Ket function (center i)
        qi (list[float]): Quadrupole integral with operator on Ket function (center i)
        dj (list[float]): Dipole integral with operator on Bra function (center j)
        qj (list[float]): Quadrupole integral with operator on Bra function (center j)
    """    

    # Create dipole operator on Bra function from Ket function and shift contribution
    # due to monopol displacement
    dj[0] += vec[0]*s
    dj[1] += vec[1]*s
    dj[2] += vec[2]*s

    # For the quadrupole operator on the Bra function we first construct the shift
    # contribution from the dipole and monopol displacement, since we have to remove
    # the trace contribution from the shift and the moment integral on the Ket function
    # is already traceless
    qj[0] = 2*vec[0]*di[0] + vec[0]**2*s
    qj[2] = 2*vec[1]*di[1] + vec[1]**2*s
    qj[5] = 2*vec[2]*di[2] + vec[2]**2*s
    qj[1] = vec[0]*di[1] + vec[1]*di[0] + vec[0]*vec[1]*s
    qj[3] = vec[0]*di[2] + vec[2]*di[0] + vec[0]*vec[2]*s
    qj[4] = vec[1]*di[2] + vec[2]*di[1] + vec[1]*vec[2]*s
    # Now collect the trace of the shift contribution
    tr = 0.5 * (qj[0] + qj[2] + qj[5])

    # Finally, assemble the quadrupole operator on the Bra function from the operator
    # on the Ket function and the traceless shift contribution
    qj[0] = qi[0] + 1.5 * qj[0] - tr
    qj[1] = qi[1] + 1.5 * qj[1]
    qj[2] = qi[2] + 1.5 * qj[2] - tr
    qj[3] = qi[3] + 1.5 * qj[3]
    qj[4] = qi[4] + 1.5 * qj[4]
    qj[5] = qi[5] + 1.5 * qj[5] - tr

    return dj, qj



def mlIdx(ml: int, l: int) -> int:
    """ Index gymnastic to transfer magnetic quantum number ordering from one convention to another """

    idx = None

    # -1, 0, +1 -> +1, -1, 0
    p = [2, 3, 1]
    # -2, -1, 0, +1, +2 -> 0, +1, -1, +2, -2
    d = [5, 3, 1, 2, 4]
    # -3, -2, -1, 0, +1, +2, +3 -> 0, +1, -1, +2, -2, 3, -3
    f = [7, 5, 3, 1, 2, 4, 6]
    # -4, -3, -2, -1, 0, +1, +2, +3, +4 -> 0, +1, -1, +2, -2, +3, -3, +4, -4
    g = [9, 7, 5, 3, 1, 2, 4, 6, 8]

    if l == 1:
        idx = p[ml]
    elif l == 2:
        idx = d[ml]
    elif l == 3:
        idx = f[ml]
    elif l == 4:
        idx = g[ml]
    else:
        idx = ml
    return idx

# The Hamiltonian is saved in an atomic block sparse compressed format.
# We calculate a shell pair as a contiguous blocks and spread it to the
# contiguous atomic block.
#
# Candidate for (partial) upstreaming in tblite library.
def buildDiatomicBlocks(iAtFirst, iAtLast, species, coords, nNeighbour, iNeighbours, img2centCell, iPair, nOrbAtom, bas, h0, selfenergy, hamiltonian, overlap, dpintBra, dpintKet, qpintBra, qpintKet):
    """[summary]

    Args:
        iAtFirst (int):  Atom range for this processor to evaluate
        iAtLast (int):  Atom range for this processor to evaluate
        species (list[int]): Chemical species of each atom
        coords (list(list[float])): Atomic coordinates
        nNeighbour (list[int]): Number of surrounding neighbours for each atom
        iNeighbours (list(list[int])): List of surrounding neighbours for each atom
        img2centCell (list[int]): Mapping of images back to atoms in the central cell
        iPair (list(list[int])): Shift vector, where the interaction between two atoms starts in the sparse format
        nOrbAtom (list[int]): Size of the block in spare format for each atom
        bas (basis_type): Basis set information
        h0 (tb_hamiltonian): Hamiltonian interaction data
        selfenergy (list[float]): Diagonal elements of the Hamiltonian
        hamiltonian (list[float]): Effective Hamiltonian
        overlap (list[float]): Overlap integral matrix
        dpintBra (list(list[float])): Dipole moment integral matrix, operator on the bra function
        dpintKet (list(list[float])): Dipole moment integral matrix, operator on the ket function
        qpintBra (list(list[float])): Quadrupole moment integral matrix, operator on the bra function
        qpintKet (list(list[float])): Quadrupole moment integral matrix, operator on the ket function
    """

    # TODO
    #type(tb_hamiltonian), intent(in) :: h0

    dtmpj = torch.zeros(dimDipole)
    qtmpj = torch.zeros(dimQuadrupole)
    stmp = torch.zeros(msao(bas.maxl)**2)
    dtmpi = torch.zeros(dimDipole, msao(bas.maxl)**2)
    qtmpi = torch.zeros(dimQuadrupole, msao(bas.maxl)**2)

    for iAt in range(iAtFirst, iAtLast):
        # TODO: do iAt = iAtFirst, iAtLast --> indexing

        iZp = species[iAt]
        iss = bas.ish_at[iAt]
        io = bas.iao_sh[iss+1]

        for iNeigh in range(nNeighbour[iAt]):

            img = iNeighbours[iNeigh, iAt]
            jAt = img2centCell[img]
            jZp = species[jAt]
            js = bas.ish_at[jAt]
            ind = iPair[iNeigh, iAt]
            jo = bas.iao_sh[js+1]
            nBlk = nOrbAtom[jAt]

            vec = coords[:, iAt] - coords[:, img] #TODO: check torch.tensor
            r2 = vec[1]**2 + vec[2]**2 + vec[3]**2
            rr = math.sqrt(math.sqrt(r2) / (h0.rad[jZp] + h0.rad[iZp]))

            for iSh in range(bas.nsh_id[iZp]):
                ii = bas.iao_sh[iss+iSh] - io
                for jSh in range(bas.nsh_id[jZp]):
                    jj = bas.iao_sh[js+jSh] - jo

                    stmp, dtmpi, qtmpi = multipole_cgto(bas.cgto[jSh, jZp], bas.cgto[iSh, iZp], r2, vec, bas.intcut, stmp, dtmpi, qtmpi)
                    #call overlap_cgto(bas.cgto[jSh, jZp], bas.cgto[iSh, iZp], &
                    #    & r2, vec, bas.intcut, stmp)

                    shpoly = (1.0 + h0.shpoly[iSh, iZp]*rr) * (1.0 + h0.shpoly[jSh, jZp]*rr)

                    hij = 0.5 * (selfenergy[iss+iSh] + selfenergy[js+jSh]) * h0.hscale[jSh, iSh, jZp, iZp] * shpoly

                    li = bas.cgto[iSh, iZp].ang
                    lj = bas.cgto[jSh, jZp].ang
                    nao = msao[lj]
                    for iao in range(msao[li]):
                        for jao in range(nao):
                            ij = mlIdx(jao, lj) + nao*(mlIdx(iao, li)-1)
                            iblk = ind + jj+jao + nBlk*(ii+iao-1)
                            
                            dtmpj, qtmpj = shift_operator(vec, stmp[ij], dtmpi[:, ij], qtmpi[:, ij], dtmpj, qtmpj)

                            overlap[iblk] = stmp[ij]

                            dpintBra[:, iblk] = dtmpj
                            dpintKet[:, iblk] = dtmpi[:, ij]

                            qpintBra[:, iblk] = qtmpj
                            qpintKet[:, iblk] = qtmpi[:, ij]

                            hamiltonian[iblk] = stmp[ij] * hij

    return overlap, dpintBra, dpintKet, qpintBra, qpintKet, hamiltonian
