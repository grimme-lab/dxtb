# This file is part of xtbml.

"""
Definition of repulsion energy terms.

"""
from typing import Optional
from pydantic import BaseModel
from torch import Tensor
import torch
from math import sqrt, exp

from tbmalt.structures.geometry import Geometry

# TODO: currently order is defined by geometry.unique_atomic_numbers()
#        maybe change to dictionary representation?

class Repulsion(BaseModel):  # TODO: inherit from Energy_Contribution
    """
    Classical repulsion interaction as used with the xTB Hamiltonian.
    Calculated is the effective repulsion as given in the TB framework.
    Container to evaluate classical repulsion interactions for the xTB Hamiltonian.
    """  # TODOC

    """Molecular structure data"""
    geometry: Geometry
    """Repulsion interaction exponent for all element pairs"""
    alpha: Optional[Tensor] = None  # shape [A, A] --> A being number of unique species
    """Effective nuclear charge for all element pairs"""
    zeff: Optional[Tensor] = None  # shape [A, A]
    """Scaling of the repulsion exponents, pairwise parameters for all element pairs"""
    kexp: Optional[Tensor] = None  # shape [A, A]
    """Exponent of the repulsion polynomial, pairwise parameters for all element pairs"""
    rexp: Optional[Tensor] = None  # shape [A, A]

    class Config:
        # allow for tensor fields
        arbitrary_types_allowed = True

    # NOTE: for optional arguments: alternatively use non-class methods> attr = Field(default_factory=_setup_attr)
    """@validator('alpha')
    def _setup_alpha(cls, alpha):
       alpha_new, l = cls._setup_generic(alpha)
       for i in range(l):
          for j in range(l):
                alpha_new[i,j] = sqrt(alpha[i]*alpha[j])
       return alpha_new

    @validator('zeff')
    def _setup_zeff(cls, zeff):
       zeff_new, l = cls._setup_generic(zeff)
       for i in range(l):
          for j in range(l):
                zeff_new[i,j] = zeff[i]*zeff[j]
       return zeff_new

    @validator('kexp')
    def _setup_kexp(cls, kexp):
       # TODO kexp will be a scalar! real(wp), intent(in) :: kexp
       #       how to setup with basemodel?
       kexp_new, l = cls._setup_generic(kexp)
       for i in range(l):
          for j in range(l):
                kexp_new[i,j] = kexp[i]*kexp[j]
       return kexp_new

    @classmethod
    def _setup_generic(cls, v):
       if len(v.shape) != 1:
          raise ValueError('shape mismatch: expect 1D')
       l = len(v) # TODO: replace with geometry.nid?
       v_new = torch.zeros((l,l))
       return v_new, l"""

    def setup(
        self, alpha: Tensor, zeff: Tensor, kexp: float, kexp_light: float, rexp: float
    ) -> None:
        """Setup internal variables.

        Raises:
           ValueError: shape mismatch for non 1D input of alpha or zeff
        """

        # NOTE: as setting up entirely in constructor is tedious in pydantic framework 
        #       (cf. e.g. https://github.com/samuelcolvin/pydantic/issues/935)
        #       this helper method is used.

        if len(alpha.shape) != 1:
            raise ValueError("shape mismatch: expect 1D")
        if len(zeff.shape) != 1:
            raise ValueError("shape mismatch: expect 1D")

        l = len(self.geometry, unique=True)
        self.alpha = torch.zeros((l, l))
        self.zeff = torch.zeros((l, l))
        self.kexp = torch.zeros((l, l))

        for i in range(l):
            for j in range(l):
                self.alpha[i, j] = sqrt(alpha[i] * alpha[j])

        for i in range(l):
            for j in range(l):
                self.zeff[i, j] = zeff[i] * zeff[j]

        for i in range(l):
            iz = self.geometry.atomic_numbers[i]
            for j in range(l):
                jz = self.geometry.atomic_numbers[j]
                self.kexp[i, j] = kexp if (iz > 2 or jz > 2) else kexp_light

        self.rexp = torch.ones((l, l)) * rexp

        return

    def get_energy(self, geometry: Geometry, trans: Tensor, cutoff: float) -> Tensor:
        """
        Obtain repulsion energy

        Args:
           geometry (Geometry): Molecular structure data
           trans (Tensor): Lattice points
           cutoff (float): Real space cutoff

        Returns:
           Tensor: Repulsion energy
        """

        n_atoms = len(geometry, unique=False)
        energies = torch.zeros(n_atoms)
        cutoff2 = cutoff ** 2

        for (iat, jat) in geometry.generate_interactions(unique=False):
            iat, jat = iat.item(), jat.item()

            # get index of atomic number in unique
            izp = geometry.get_species_index(geometry.atomic_numbers[iat]) 
            jzp = geometry.get_species_index(geometry.atomic_numbers[jat]) 

            for itr in range(trans.shape[0]):
                    rij = (
                        geometry.positions[iat, :]
                        - geometry.positions[jat, :]
                        - trans[itr, :] # TODO: check correct shape of translation
                    )
                    r2 = sum(rij ** 2)
                    if r2 > cutoff2 or r2 < 1.0e-12: # TODO: for eg CH4 (x1=-x2; y1=-y2) get contributions correct
                        continue
                    r1 = torch.sqrt(r2)

                    # TODO: get correct values (check storage structure)
                    r1k = r1 ** self.kexp[izp, jzp]
                    exa = exp(-self.alpha[izp, jzp] * r1k)
                    r1r = r1 ** self.rexp[izp, jzp]

                    dE = self.zeff[izp, jzp] * exa / r1r

                    # partition energy equally on contributing atoms
                    energies[iat] = energies[iat] + 0.5 * dE
                    if iat != jat:
                        energies[jat] = energies[jat] + 0.5 * dE

        return sum(energies)

    def get_gradient(self, geometry: Geometry, trans: Tensor, cutoff: float) -> Tensor:
        """
        Obtain repulsion gradient and energy.

        Args:
           geometry (Geometry): Molecular structure data
           trans (Tensor): Lattice points
           cutoff (float): Real space cutoff

        Returns:
           Tensor: Repulsion energy
        """

        n_atoms = len(geometry, unique=False)
        energies = torch.zeros(n_atoms)
        cutoff2 = cutoff ** 2

        # Molecular gradient of the repulsion energy
        gradient = torch.zeros((n_atoms, 3))
        # Strain derivatives of the repulsion energy
        sigma = torch.zeros((3, 3))
        dG = torch.zeros((3))
        dS = torch.zeros((3, 3))

        # TODO: apply same adjustments as for energy

        for (iat, jat) in geometry.generate_interactions(unique=False):
            iat, jat = iat.item(), jat.item()

            # get index of atomic number in unique
            izp = geometry.get_species_index(geometry.atomic_numbers[iat]) 
            jzp = geometry.get_species_index(geometry.atomic_numbers[jat]) 

            for itr in range(trans.shape[0]):
                  rij = (
                        geometry.positions[iat, :]
                        - geometry.positions[jat, :]
                        - trans[itr, :]
                  )
                  r2 = sum(rij ** 2)
                  if r2 > cutoff2 or r2 < 1.0e-12:
                        continue
                  r1 = torch.sqrt(r2)

                  r1k = r1 ** self.kexp[izp, jzp]
                  exa = exp(-self.alpha[izp, jzp] * r1k)
                  r1r = r1 ** self.rexp[izp, jzp]

                  dE = self.zeff[izp, jzp] * exa / r1r

                  dG = (
                     -(
                           self.alpha[izp, jzp] * r1k * self.kexp[izp, jzp]
                           + self.rexp[izp, jzp]
                     )
                     *dE* rij/r2
                  )
                  dS = dG.repeat(3, 1) * rij.repeat(
                     3, 1
                  )  # spread(dG, 1, 3) * spread(rij, 2, 3)

                  # partition energy and gradient equally on contributing atoms
                  energies[iat] = energies[iat] + 0.5 * dE
                  if iat != jat:
                     energies[jat] = energies[jat] + 0.5 * dE
                     gradient[iat, :] = gradient[iat, :] + dG
                     gradient[jat, :] = gradient[jat, :] - dG
                     sigma = sigma + dS
                  else:
                     sigma += 0.5 * dS

        return gradient, sigma, energies

    def get_virial(self) -> Tensor:
        """
        Obtain repulsion virial
        """
        raise NotImplementedError
