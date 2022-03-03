# This file is part of xtbml.

"""
Definition of repulsion energy terms.

"""

from typing import Optional, Union, Tuple
from torch import Tensor
import torch
from math import sqrt, exp

from tbmalt.structures.geometry import Geometry

from .base import Energy_Contribution

class Repulsion(Energy_Contribution):
    """
    Classical repulsion interaction as used with the xTB Hamiltonian.
    Calculated is the effective repulsion as given in the TB framework.
    Container to evaluate classical repulsion interactions for the xTB Hamiltonian.
    """

    """Repulsion interaction exponent for all element pairs"""
    alpha: Optional[Tensor] = None  # shape [A, A] --> A being number of unique species
    """Effective nuclear charge for all element pairs"""
    zeff: Optional[Tensor] = None  # shape [A, A]
    """Scaling of the repulsion exponents, pairwise parameters for all element pairs"""
    kexp: Optional[Tensor] = None  # shape [A, A]
    """Exponent of the repulsion polynomial, pairwise parameters for all element pairs"""
    rexp: Optional[Tensor] = None  # shape [A, A]

    def setup(
        self, alpha: Tensor, zeff: Tensor, kexp: float, kexp_light: float, rexp: float
    ) -> None:
        """Setup internal variables.

        Raises:
           ValueError: shape mismatch for non 1D input of alpha or zeff
        """

        if len(alpha.shape) != 1:
            raise ValueError("shape mismatch: expect 1D")
        if len(zeff.shape) != 1:
            raise ValueError("shape mismatch: expect 1D")

        l = self.geometry.get_length(unique=True)
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

    def get_engrad(self, geometry: Geometry, cutoff: float, calc_gradient: bool = False) -> Union[Tensor, Tuple[Tensor,Tensor]]:
        """
        Obtain repulsion energy and gradient.

        Args:
           geometry (Geometry): Molecular structure data
           cutoff (float): Real space cutoff

        Returns:
           Tensor: Repulsion energy and gradient
        """

        n_atoms = geometry.get_length(unique=False)
        energies = torch.zeros(n_atoms)
        cutoff2 = cutoff ** 2
        
        if calc_gradient:
            # Molecular gradient of the repulsion energy
            gradient = torch.zeros((n_atoms, 3))

        for (iat, jat) in geometry.generate_interactions(unique=False):
            iat, jat = iat.item(), jat.item()

            # get index of atomic number in unique
            izp = geometry.get_species_index(geometry.atomic_numbers[iat]) 
            jzp = geometry.get_species_index(geometry.atomic_numbers[jat]) 

            rij = (
                  geometry.positions[iat, :]
                  - geometry.positions[jat, :]
            )
            r2 = sum(rij ** 2)
            if r2 > cutoff2 or r2 < 1.0e-12:
                  continue
            r1 = torch.sqrt(r2)

            r1k = r1 ** self.kexp[izp, jzp]
            exa = exp(-self.alpha[izp, jzp] * r1k)
            r1r = r1 ** self.rexp[izp, jzp]

            dE = self.zeff[izp, jzp] * exa / r1r

            if calc_gradient:
                dG = (
                -(self.alpha[izp, jzp] * r1k * self.kexp[izp, jzp] + self.rexp[izp, jzp])
                *dE* rij/r2
                )

            # partition energy and gradient equally on contributing atoms
            energies[iat] = energies[iat] + 0.5 * dE
            if calc_gradient:
                gradient[iat, :] = gradient[iat, :] + dG

        energies = torch.sum(energies)

        if calc_gradient:
            return energies, gradient
        else:
            return energies

