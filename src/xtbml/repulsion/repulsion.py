# This file is part of xtbml.

"""
Definition of repulsion energy terms.

"""

from typing import Dict, Optional, Union, Tuple
from torch import Tensor
import torch
from math import sqrt, exp

from xtbml.constants.torch import FLOAT64
from xtbml.exlibs.tbmalt import Geometry

from .base import Energy_Contribution


torch.set_default_dtype(FLOAT64)  # required for repulsion tests (esp. gradients)


class Repulsion(Energy_Contribution):
    """
    Classical repulsion interaction as used with the xTB Hamiltonian.
    Calculated is the effective repulsion as given in the TB framework.
    Container to evaluate classical repulsion interactions for the xTB Hamiltonian.
    """

    alpha: Optional[Tensor] = None  # shape [87]
    """Repulsion interaction exponent for all element pairs"""

    zeff: Optional[Tensor] = None  # shape [87]
    """Effective nuclear charge for all element pairs"""

    kexp: Optional[Tensor] = None  # shape [87]
    """Scaling of the repulsion exponents, pairwise parameters for all element pairs"""

    def get_kexp_as_tensor(self, gfn_repulsion):
        tnsr = torch.ones(87) * gfn_repulsion.kexp

        # dummy for indexing with atomic numbers
        tnsr[0] = -999999

        # change value for H and He
        if gfn_repulsion.kexp_light is not None:
            tnsr[1:3] = gfn_repulsion.kexp_light

        return tnsr

    def setup(
        self,
        alpha: Tensor,
        zeff: Tensor,
        kexp_data: Dict[str, Union[float, None]],
    ) -> None:
        """Setup internal variables."""

        numbers = self.geometry.atomic_numbers

        # mask for padding
        real = numbers != 0
        mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))

        # set diagonal to 0 to remove A=B case in summation
        torch.diagonal(mask, dim1=-2, dim2=-1)[:] = True

        # create padded array
        self.alpha = torch.sqrt(
            alpha[numbers].unsqueeze(-2) * alpha[numbers].unsqueeze(-1)
        )
        self.zeff = zeff[numbers].unsqueeze(-2) * zeff[numbers].unsqueeze(-1)

        kexp = self.get_kexp_as_tensor(kexp_data)
        self.kexp = kexp[numbers].unsqueeze(-2) * kexp.new_ones(kexp.shape)[
            numbers
        ].unsqueeze(-1)

        self.alpha[mask] = 0
        self.zeff[mask] = 0
        self.kexp[mask] = 0

        # OLD CODE
        # if len(alpha.shape) != 1:
        #     raise ValueError("shape mismatch: expect 1D")
        # if len(zeff.shape) != 1:
        #     raise ValueError("shape mismatch: expect 1D")

        # l = self.geometry.get_length(unique=True)
        # self.alpha = torch.zeros((l, l))
        # self.zeff = torch.zeros((l, l))
        # self.kexp = torch.zeros((l, l))

        # for i in range(l):
        #     for j in range(l):
        #         self.alpha[i, j] = sqrt(alpha[i] * alpha[j])

        # for i in range(l):
        #     for j in range(l):
        #         self.zeff[i, j] = zeff[i] * zeff[j]

        # for i in range(l):
        #     iz = self.geometry.unique_atomic_numbers()[i]
        #     for j in range(l):
        #         jz = self.geometry.unique_atomic_numbers()[j]
        #         self.kexp[i, j] = kexp if (iz > 2 or jz > 2) else kexp_light

        # self.rexp = torch.ones(l, l) * rexp

    def get_engrad(
        self, geometry: Geometry, cutoff: float, calc_gradient: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Obtain repulsion energy and gradient.

        Args:
           geometry (Geometry): Molecular structure data
           cutoff (float): Real space cutoff

        Returns:
           Tensor: Repulsion energy and gradient
        """

        # TODO: implement cutoff
        # cutoff2 = cutoff**2

        # FIXME?: add epsilon to avoid zero division in dE term (gives inf)
        # -> in ncoord problem is mitigated by exp(-inf) = 0
        distances = geometry.distances + 2.220446049250313e-16

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(distances, self.kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-self.alpha * r1k)

        # Eq.13: repulsion energy
        dE = self.zeff * exp_term / distances

        # Eq.13: sum up and rectify double counting (symmetric matrix)
        s = 0.5 * torch.sum(dE, dim=(-2, -1))

        # TODO
        if calc_gradient:
            rij_r2 = 1
            # dG_vec = -(self.alpha * r1k * self.kexp + REXP) * dE * rij_r2
            # print("grad vec", dG_vec)
            return s, s
        else:
            return s

        # OLD CODE
        # for (iat, jat) in geometry.generate_interactions(unique=False):
        #     iat, jat = iat.item(), jat.item()

        #     # geometry accessed by atom index
        #     rij = geometry.positions[iat, :] - geometry.positions[jat, :]

        #     # translate atom index to unique species index
        #     # (== index of atomic number in unique)
        #     idx_i = geometry.get_species_index(geometry.atomic_numbers[iat])
        #     idx_j = geometry.get_species_index(geometry.atomic_numbers[jat])

        #     # distances within cutoff
        #     r2 = sum(rij**2)
        #     if r2 > cutoff2 or r2 < 1.0e-12:
        #         continue
        #     r1 = torch.sqrt(r2)

        #     # R_AB ** k_f
        #     r1k = r1 ** self.kexp[idx_i, idx_j]

        #     # exp(- (alpha_A * alpha_B)**2 )
        #     exa = exp(-self.alpha[idx_i, idx_j] * r1k)

        #     # ????
        #     r1r = r1 ** self.rexp[idx_i, idx_j]

        #     dE = self.zeff[idx_i, idx_j] * exa / r1r

        #     if calc_gradient:
        #         dG = (
        #             -(
        #                 self.alpha[idx_i, idx_j] * r1k * self.kexp[idx_i, idx_j]
        #                 + self.rexp[idx_i, idx_j]
        #             )
        #             * dE
        #             * rij
        #             / r2
        #         )

        #     # partition energy and gradient equally on contributing atoms
        #     energies[iat] = energies[iat] + 0.5 * dE

        #     if iat != jat:
        #         energies[jat] = energies[jat] + 0.5 * dE

        #         if calc_gradient:
        #             gradient[iat, :] = gradient[iat, :] + dG
        #             gradient[jat, :] = gradient[jat, :] - dG
        #     else:
        #         # should never happen, since iat==jat --> r2==0.0
        #         # NOTE: only for PBC with transition vector this might be raised
        #         raise ValueError

        # energies = torch.sum(energies)

        # if calc_gradient:
        #     print("final gradient", gradient)
        #     return energies, gradient
        # else:
        #     return energies
