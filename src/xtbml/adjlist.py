import torch
from numpy import finfo

from xtbml.constants.torch import UINT8 as DTYPE


class AdjacencyList:
    def __init__(self, mol, trans, cutoff):
        self.trans = trans
        self.inl = torch.zeros(mol.get_length(), dtype=DTYPE)
        self.nnl = torch.zeros(mol.get_length(), dtype=DTYPE)

        tmp_nlat = torch.zeros(10 * mol.get_length(), dtype=DTYPE)
        tmp_nltr = torch.zeros(10 * mol.get_length(), dtype=DTYPE)

        img = 0
        cutoff2 = cutoff * cutoff

        for iat in range(mol.get_length()):
            self.inl[iat] = img

            for jat in range(iat + 1):
                for itr in range(trans.size(dim=1)):
                    vec = torch.sub(mol.positions[iat, :], mol.positions[jat, :])
                    vec = torch.sub(vec, trans[itr, :])
                    r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]
                    if r2 < finfo(type(cutoff2)).eps or r2 > cutoff2:
                        continue

                    tmp_nlat[img] = jat
                    tmp_nltr[img] = itr
                    img += 1

            self.nnl[iat] = img - self.inl[iat]

        self.nlat = tmp_nlat[:img]
        self.nltr = tmp_nltr[:img]
