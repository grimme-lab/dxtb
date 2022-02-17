
import math
import torch

""" Definition of basic data classes """


# Maximum contraction length of basis functions.
# The limit is chosen as twice the maximum size returned by the STO-NG expansion
maxg = 12

class Cgto_Type():
    """  Contracted Gaussian type basis function """

    def __init__(self):
        # Angular momentum of this basis function
        self.ang = -1
        # Contraction length of this basis function
        self.nprim = 0
        # Exponent of the primitive Gaussian functions
        self.alpha = torch.tensor([0.0 for _ in range(maxg)])
        # Contraction coefficients of the primitive Gaussian functions,
        # might contain normalization
        self.coeff = torch.tensor([0.0 for _ in range(maxg)])

    def __str__(self):
        return f"cgto( l:{self.ang} | ng:{self.nprim} | alpha:{self.alpha} | coeff:{self.coeff} )"

class Basis_Type():
    """ Collection of information regarding the basis set of a system """

    def __init__(self) -> None:

        # Maximum angular momentum of all basis functions,
        # used to determine scratch size in integral calculation
        self.maxl = 0
        # Number of shells in this basis set
        self.nsh = 0
        # Number of spherical atomic orbitals in this basis set
        self.nao = 0
        # Integral cutoff as maximum exponent of Gaussian product theoreom to consider
        self.intcut = 0.0
        # Smallest primitive exponent in the basis set
        self.min_alpha = float("inf")
        # Number of shells for each species
        self.nsh_id = torch.tensor([])
        # Number of shells for each atom
        self.nsh_at = torch.tensor([])
        # Number of spherical atomic orbitals for each shell
        self.nao_sh = torch.tensor([])
        # Index offset for each shell in the atomic orbital space
        self.iao_sh = torch.tensor([])
        # Index offset for each atom in the shell space
        self.ish_at = torch.tensor([])
        # Mapping from spherical atomic orbitals to the respective atom
        self.ao2at = torch.tensor([])
        # Mapping from spherical atomic orbitals to the respective shell
        self.ao2sh = torch.tensor([])
        # Mapping from shells to the respective atom
        self.sh2at = torch.tensor([])
        # Contracted Gaussian basis functions forming the basis set
        self.cgto = Cgto_Type()


    def new_basis(self, mol, nshell, cgto, acc):
        """ Create a new basis set
        Args:
            mol (structure_type): Molecular structure data
            nshell (List(int)): Number of shells per species
            cgto (List(List(Cgto_Type))): Contracted Gaussian basis functions for each shell and species
            acc (float): Calculation accuracy

        Returns:
            Basis_Type: new basis set
        """  

        self.nsh_id = nshell
        self.cgto = cgto
        self.intcut = integral_cutoff(acc) # TODO

        # Make count of shells for each atom
        self.nsh_at = nshell[mol.id]

        # Create mapping between atoms and shells
        self.nsh = sum(self.nsh_at)
        ii = 0
        for iat in range(1, mol.nat):
            self.ish_at[iat] = ii
            for ish in range(1, self.nsh_at[iat]):
                self.sh2at[ii+ish] = iat
            ii += self.nsh_at[iat]

        # Make count of spherical orbitals for each shell
        for iat in range(1, mol.nat):
            isp = mol.id[iat]
            ii = self.ish_at[iat]
            for ish in range(1, self.nsh_at[iat]):
                self.nao_sh[ii+ish] = 2*cgto[ish, isp].ang + 1

        # Create mapping between shells and spherical orbitals, also map directly back to atoms
        self.nao = sum(self.nao_sh)
        ii = 0
        for ish in range(1, self.nsh):
            self.iao_sh[ish] = ii
            for iao in range(1, self.nao_sh[ish]):
                self.ao2sh[ii+iao] = ish
                self.ao2at[ii+iao] = self.sh2at[ish]
            ii += self.nao_sh[ish]

        ii = 0
        for iat in range(1, mol.nat):
            isp = mol.id[iat]
            for ish in range(1, nshell[isp]):
                self.iao_sh[ish+self.ish_at[iat]] = ii
                ii = ii + 2*cgto(ish, isp).ang + 1

        min_alpha = float("inf")

        for isp in range(1, len(nshell)):
            for ish in range(1, nshell[isp]):
                self.maxl = max(self.maxl, cgto[ish, isp].ang)
                min_alpha = min(min_alpha, min(cgto[ish, isp].alpha[:cgto[ish, isp].nprim]))

        self.min_alpha = min_alpha

        return self

    def get_cutoff(self, acc=None):
        """ Determine required real space cutoff for the basis set. 
            Get optimal real space cutoff for integral evaluation

        Args:
            acc (float, optional): Accuracy for the integral cutoff. Defaults to None.

        Returns:
            float: Required realspace cutoff
        """
        max_cutoff = 40.0
        if acc is not None:
            intcut = integral_cutoff(acc)
        else:
            intcut = self.intcut
        # ai * aj * cutoff2 / (ai + aj) == intcut
        cutoff = min(math.sqrt(2.0*intcut/self.min_alpha), max_cutoff)
        return cutoff

def integral_cutoff(acc=None):
    """ Create integral cutoff from accuracy value

    Args:
        acc (float, optional): Accuracy for the integral cutoff. Defaults to None.

    Returns:
        float: Integral cutoff
    """
    min_intcut = 5.0
    max_intcut = 25.0
    max_acc = 1.0e-4
    min_acc = 1.0e+3
    intcut = clip(max_intcut - 10*math.log10(clip(acc, min_acc, max_acc)), min_intcut, max_intcut)
    return intcut

def clip(val: float, min_val: float, max_val: float):
    return min(max(val, min_val), max_val)




