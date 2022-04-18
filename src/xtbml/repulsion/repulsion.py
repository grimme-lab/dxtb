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
from xtbml.param.base import Param
from xtbml.param.element import Element
from xtbml.param.repulsion import EffectiveRepulsion

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

    def get_kexp(self, par_repulsion: EffectiveRepulsion):
        tnsr = torch.ones(87) * par_repulsion.kexp

        # dummy for indexing with atomic numbers
        tnsr[0] = -999999

        # change value for H and He
        if par_repulsion.kexp_light is not None:
            tnsr[1:3] = par_repulsion.kexp_light

        return tnsr

    def get_alpha(self, par: Dict[str, Element]) -> Tensor:
        a = torch.zeros(87)

        # dummy for indexing with atomic numbers
        a[0] = -999999

        for i, item in enumerate(par.values()):
            a[i + 1] = item.arep

        return a

    def get_zeff(self, par: Dict[str, Element]) -> Tensor:
        z = torch.zeros(87)

        # dummy for indexing with atomic numbers
        z[0] = -999999

        for i, item in enumerate(par.values()):
            z[i + 1] = item.zeff

        return z

    def setup(
        self, par_element: Dict[str, Element], par_repulsion: EffectiveRepulsion
    ) -> None:
        """Setup internal variables."""

        alpha = self.get_alpha(par_element)
        zeff = self.get_zeff(par_element)
        kexp = self.get_kexp(par_repulsion)

        numbers = self.geometry.atomic_numbers

        # mask for padding
        real = numbers != 0
        mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))

        # set diagonal to 0 to remove A=B case in summation
        torch.diagonal(mask, dim1=-2, dim2=-1)[:] = True

        # create padded arrays
        self.alpha = torch.sqrt(
            alpha[numbers].unsqueeze(-2) * alpha[numbers].unsqueeze(-1)
        )
        self.zeff = zeff[numbers].unsqueeze(-2) * zeff[numbers].unsqueeze(-1)

        self.kexp = kexp[numbers].unsqueeze(-2) * kexp.new_ones(kexp.shape)[
            numbers
        ].unsqueeze(-1)

        self.alpha[mask] = 0
        self.zeff[mask] = 0
        self.kexp[mask] = 0

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

        # FIXME?: add epsilon to avoid zero division in some terms (gives inf)
        # -> in ncoord problem is mitigated by exp(-inf) = 0
        EPS = 2.220446049250313e-16

        distances = geometry.distances

        # Calculate repulsion only for distances smaller than cutoff
        if cutoff is not None:
            distances = torch.where(distances <= cutoff, distances, 0.0)

        distances += EPS

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(distances, self.kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-self.alpha * r1k)

        # Eq.13: repulsion energy
        dE = self.zeff * exp_term / distances

        # Eq.13: sum up and rectify double counting (symmetric matrix)
        s = 0.5 * torch.sum(dE, dim=(-2, -1))

        if calc_gradient:
            # dG.shape: (n_atoms, n_atoms)
            dG = -(self.alpha * r1k * self.kexp + 1.0) * dE

            rij = geometry.distance_vectors
            r2 = torch.sum(torch.pow(rij, 2), dim=-1, keepdim=True) + EPS

            # dG.shape: (n_atoms, n_atoms, 3)
            dG = dG.unsqueeze(-1) * rij / r2

            # dG.shape: (n_atoms, 3)
            dG = torch.sum(dG, dim=-2)

            return s, dG
        else:
            return s
