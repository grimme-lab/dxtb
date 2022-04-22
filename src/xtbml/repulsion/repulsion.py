# This file is part of xtbml.

"""Definition of repulsion energy terms."""

from typing import Dict, List, Optional, Union, Tuple
import torch
from torch import Tensor

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

    dummy_value: float = 0
    """
    Dummy value for zero index of tensors indexed by atomic numbers.
    """

    alpha: Optional[Tensor] = None
    """
    Repulsion interaction exponent for all element pairs. Always has size `RepulsionFactory.n_elements`.
    """

    w_alpha: Optional[Tensor] = None
    """
    Working tensor for repulsion interaction exponent for all element pairs. Size is equal to the number of atoms.
    """

    zeff: Optional[Tensor] = None
    """
    Effective nuclear charge for all element pairs. Always has size `RepulsionFactory.n_elements`.
    """

    w_zeff: Optional[Tensor] = None
    """
    Working tensor for effective nuclear charge for all element pairs. Size is equal to the number of atoms.
    """

    kexp: Optional[Tensor] = None
    """
    Scaling of the repulsion exponents. Always has size `RepulsionFactory.n_elements`.
    """

    w_kexp: Optional[Tensor] = None
    """
    Working tensor for scaling of the repulsion exponents. Size is equal to the number of atoms.
    """

    def get_kexp(self, par_repulsion: EffectiveRepulsion) -> List[float]:
        """Obtain exponential scaling factors of effective repulsion.

        Args:
            par_repulsion (EffectiveRepulsion): Repulsion parametrization.

        Returns:
            Tensor: Exponential scaling factors of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """

        kexp = [par_repulsion.kexp] * self.n_elements

        # dummy for indexing with atomic numbers
        kexp[0] = self.dummy_value

        # change value for H and He
        if par_repulsion.kexp_light is not None:
            kexp[1] = par_repulsion.kexp_light
            kexp[2] = par_repulsion.kexp_light

        return kexp

    def get_alpha(self, par: Dict[str, Element]) -> List[float]:
        """Obtain alpha parameters as tensors.

        Args:
            par (Dict[str, Element]): Parametrization of elements.

        Returns:
            Tensor: Alpha parameter of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """

        # dummy for indexing with atomic numbers
        a = [self.dummy_value]

        for item in par.values():
            a.append(item.arep)

        return a

    def get_zeff(self, par: Dict[str, Element]) -> List[float]:
        """Obtain effective charges as tensors.

        Args:
            par (Dict[str, Element]): Parametrization of elements.

        Returns:
            Tensor: Effective charges of all elements (with 0 index being a dummy to allow indexing by atomic numbers).
        """

        # dummy for indexing with atomic numbers
        z = [self.dummy_value]

        for item in par.values():
            z.append(item.zeff)

        return z

    def setup(
        self, par_element: Dict[str, Element], par_repulsion: EffectiveRepulsion
    ) -> None:
        """Setup internal variables.

        Args:
            par_element (Dict[str, Element]): Parametrization of elements.
            par_repulsion (EffectiveRepulsion): Parametrization of repulsion.
        """

        if self.req_grad is None:
            self.req_grad = False

        numbers = self.numbers
        dtype = self.positions.dtype

        # get parameters and format to tensors
        self.alpha = torch.tensor(
            self.get_alpha(par_element), dtype=dtype, requires_grad=self.req_grad
        )
        alpha_mol = self.alpha[numbers]

        self.zeff = torch.tensor(
            self.get_zeff(par_element), dtype=dtype, requires_grad=self.req_grad
        )
        zeff_mol = self.zeff[numbers]

        self.kexp = torch.tensor(
            self.get_kexp(par_repulsion), dtype=dtype, requires_grad=self.req_grad
        )
        kexp_mol = self.kexp[numbers]

        # mask for padding, inverted for multiplication
        real = numbers != 0
        mask = real.unsqueeze(-2) * real.unsqueeze(-1)

        # set diagonal to 0 to remove A=B case in summation
        mask.diagonal(dim1=-2, dim2=-1).fill_(False)

        # create padded arrays and write to working tensor
        alpha_mul = alpha_mol.unsqueeze(-1) * alpha_mol.unsqueeze(-2)
        self.w_alpha = torch.sqrt(alpha_mul) * mask

        zeff_mul = zeff_mol.unsqueeze(-2) * zeff_mol.unsqueeze(-1)
        self.w_zeff = zeff_mul * mask

        kexp_mul = kexp_mol.unsqueeze(-2) * kexp_mol.new_ones(kexp_mol.shape).unsqueeze(
            -1
        )
        self.w_kexp = kexp_mul * mask

    def get_engrad(
        self, cutoff: Optional[float] = None, calc_gradient: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Obtain repulsion energy and gradient.

        Args:
            geometry (Geometry): Structure data.
            cutoff (float): Real space cutoff.
            calc_gradient (bool, optional): Flag for calculating gradient. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: Repulsion energy (and gradient).
        """

        if self.w_alpha is None or self.w_zeff is None or self.w_kexp is None:
            raise ValueError("Working tensor is not initialized.")

        real = self.numbers != 0
        mask = ~(real.unsqueeze(-2) * real.unsqueeze(-1))

        distances = torch.cdist(self.positions, self.positions, p=2)
        distances[mask] = 0

        # Calculate repulsion only for distances smaller than cutoff
        if cutoff is not None:
            zero = distances.new_zeros(1)
            distances = torch.where(distances <= cutoff, distances, zero)

        # add epsilon to avoid zero division in some terms
        distances += torch.finfo(distances.dtype).eps

        # Eq.13: R_AB ** k_f
        r1k = torch.pow(distances, self.w_kexp)

        # Eq.13: exp(- (alpha_A * alpha_B)**0.5 * R_AB ** k_f )
        exp_term = torch.exp(-self.w_alpha * r1k)

        # Eq.13: repulsion energy
        dE = self.w_zeff * exp_term / distances

        # Eq.13: sum up and rectify double counting (symmetric matrix)
        sum_dE = 0.5 * torch.sum(dE, dim=(-2, -1))

        if calc_gradient is True:
            dG = -(self.w_alpha * r1k * self.w_kexp + 1.0) * dE
            # >>> print(dG.shape)
            # torch.Size([n_batch, n_atoms, n_atoms])

            rij = self.positions.unsqueeze(-2) - self.positions.unsqueeze(-3)
            rij[mask] = 0
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
