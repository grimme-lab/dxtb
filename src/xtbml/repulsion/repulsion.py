# This file is part of xtbml.

"""Definition of repulsion energy terms."""

from typing import Dict, Optional, Union, Tuple
import torch
from torch import Tensor

from ..exlibs.tbmalt import Geometry
from ..param import Element, EffectiveRepulsion
from .base import EnergyContribution


class RepulsionFactory(EnergyContribution):
    """
    Classical repulsion interaction as used with the xTB Hamiltonian.
    Calculated is the effective repulsion as given in the TB framework.
    Container to evaluate classical repulsion interactions for the xTB Hamiltonian.

    For details, see GFN1-xTB paper:
    Grimme, S.; Bannwarth, C.; Shushkov, P. A Robust and Accurate Tight-Binding Quantum Chemical Method for Structures, Vibrational Frequencies, and Noncovalent Interactions of Large Molecular Systems Parameterized for All spd-Block Elements (Z = 1-86). J. Chem. Theory Comput. 2017, 13 (5), 1989-2009 DOI: 10.1021/acs.jctc.7b00118
    """

    n_elements: int = 87
    """
    Number of elements (86, PSE up to Rn) plus dummy to allow indexing by atomic numbers.
    """

    dummy_value: int = -999999
    """
    Dummy value for zero index of tensors indexed by atomic numbers.
    """

    alpha: Optional[Tensor] = None
    """
    Repulsion interaction exponent for all element pairs. Always has size `RepulsionFactory.n_elements`.
    """

    zeff: Optional[Tensor] = None
    """
    Effective nuclear charge for all element pairs. Always has size `RepulsionFactory.n_elements`.
    """

    kexp: Optional[Tensor] = None
    """
    Scaling of the repulsion exponents, pairwise parameters for all element pairs. Always has size `RepulsionFactory.n_elements`.
    """

    def get_kexp(self, par_repulsion: EffectiveRepulsion) -> Tensor:
        """Obtain exponential scaling factors of effective repulsion.

        Args:
            par_repulsion (EffectiveRepulsion): Repulsion parametrization.

        Returns:
            Tensor: Exponential scaling factors of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """
        tnsr = torch.ones(self.n_elements) * par_repulsion.kexp

        # dummy for indexing with atomic numbers
        tnsr[0] = self.dummy_value

        # change value for H and He
        if par_repulsion.kexp_light is not None:
            tnsr[1:3] = par_repulsion.kexp_light

        return tnsr

    def get_alpha(self, par: Dict[str, Element]) -> Tensor:
        """Obtain alpha parameters as tensors.

        Args:
            par (Dict[str, Element]): Parametrization of elements.

        Returns:
            Tensor: Alpha parameter of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """
        a = torch.zeros(self.n_elements)

        # dummy for indexing with atomic numbers
        a[0] = self.dummy_value

        for i, item in enumerate(par.values()):
            a[i + 1] = item.arep

        return a

    def get_zeff(self, par: Dict[str, Element]) -> Tensor:
        """Obtain effective charges as tensors.

        Args:
            par (Dict[str, Element]): Parametrization of elements.

        Returns:
            Tensor: Effective charges of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """
        z = torch.zeros(self.n_elements)

        # dummy for indexing with atomic numbers
        z[0] = self.dummy_value

        for i, item in enumerate(par.values()):
            z[i + 1] = item.zeff

        return z

    def setup(
        self, par_element: Dict[str, Element], par_repulsion: EffectiveRepulsion
    ) -> None:
        """Setup internal variables.

        Args:
            par_element (Dict[str, Element]): Parametrization of elements.
            par_repulsion (EffectiveRepulsion): Parametrization of repulsion.
        """

        # get parameters and format to tensors
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
        self.alpha[mask] = 0

        self.zeff = zeff[numbers].unsqueeze(-2) * zeff[numbers].unsqueeze(-1)
        self.zeff[mask] = 0

        self.kexp = kexp[numbers].unsqueeze(-2) * kexp.new_ones(kexp.shape)[
            numbers
        ].unsqueeze(-1)
        self.kexp[mask] = 0

    def get_engrad(
        self, geometry: Geometry, cutoff: float, calc_gradient: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Obtain repulsion energy and gradient.

        Args:
            geometry (Geometry): Structure data.
            cutoff (float): Real space cutoff.
            calc_gradient (bool, optional): Flag for calculating gradient. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Repulsion energy (and gradient).
        """

        distances = geometry.distances

        # Calculate repulsion only for distances smaller than cutoff
        if cutoff is not None:
            zero = distances.new_zeros(1)
            distances = torch.where(distances <= cutoff, distances, zero)

        # add epsilon to avoid zero division in some terms
        distances += torch.finfo(geometry.dtype).eps

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(distances, self.kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-self.alpha * r1k)

        # Eq.13: repulsion energy
        dE = self.zeff * exp_term / distances

        # Eq.13: sum up and rectify double counting (symmetric matrix)
        sum_dE = 0.5 * torch.sum(dE, dim=(-2, -1))

        if calc_gradient is True:
            dG = -(self.alpha * r1k * self.kexp + 1.0) * dE
            # >>> print(dG.shape)
            # torch.Size([n_batch, n_atoms, n_atoms])

            rij = geometry.distance_vectors
            # >>> print(rij.shape)
            # torch.Size([n_batch, n_atoms, n_atoms, 3])
            r2 = torch.pow(distances, 2)
            # >>> print(r2.shape)
            # torch.Size([n_batch, n_atoms, n_atoms])

            dG = dG / r2
            dG = dG.unsqueeze(-1) * rij
            # >>> print(dG.shape)
            # torch.Size([n_batch, n_atoms, n_atoms, 3])

            dG = torch.sum(dG, dim=-2)
            # >>> print(dG.shape)
            # torch.Size([n_batch, n_atoms, 3])

            return sum_dE, dG
        else:
            return sum_dE
