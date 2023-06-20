"""
Isotropic second-order electrostatic energy (ES2)
=================================================

This module implements the second-order electrostatic energy for GFN1-xTB.

Example
-------
>>> import torch
>>> import xtbml.coulomb.secondorder as es2
>>> from xtbml.coulomb.average import harmonic_average as average
>>> from xtbml.param import GFN1_XTB, get_element_param
>>> numbers = torch.tensor([14, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [0.00000000000000, -0.00000000000000, 0.00000000000000],
...     [1.61768389755830, 1.61768389755830, -1.61768389755830],
...     [-1.61768389755830, -1.61768389755830, -1.61768389755830],
...     [1.61768389755830, -1.61768389755830, 1.61768389755830],
...     [-1.61768389755830, 1.61768389755830, 1.61768389755830],
... ])
>>> q = torch.tensor([
...     -8.41282505804719e-2,
...     2.10320626451180e-2,
...     2.10320626451178e-2,
...     2.10320626451179e-2,
...     2.10320626451179e-2,
... ])
>>> # get parametrization
>>> gexp = torch.tensor(GFN1_XTB.charge.effective.gexp)
>>> hubbard = get_element_param(GFN1_XTB.element, "gam")
>>> # calculate energy
>>> es = es2.ES2(hubbard=hubbard, average=average, gexp=gexp)
>>> cache = es.get_cache(numbers, positions)
>>> e = es.get_energy(qat, cache)
>>> torch.set_printoptions(precision=7)
>>> print(torch.sum(e, dim=-1))
tensor(0.0005078)
"""
from __future__ import annotations

import torch

from .._types import Slicers, Tensor, TensorLike
from ..basis import IndexHelper
from ..constants import xtb
from ..interaction import Interaction
from ..param import Param, get_elem_param
from ..utils import batch, real_pairs
from .average import AveragingFunction, averaging_function, harmonic_average

__all__ = ["ES2", "new_es2"]


class ES2(Interaction):
    """
    Isotropic second-order electrostatic energy (ES2).
    """

    hubbard: Tensor
    """Hubbard parameters of all elements."""

    lhubbard: Tensor | None
    """
    Shell-resolved scaling factors for Hubbard parameters (default: `None`,
    i.e., no shell resolution).
    """

    average: AveragingFunction
    """
    Function to use for averaging the Hubbard parameters (default:
    `~dxtb.coulomb.average.harmonic_average`).
    """

    gexp: Tensor
    """Exponent of the second-order Coulomb interaction (default: 2.0)."""

    shell_resolved: bool
    """Electrostatics is shell-resolved (default: `True`)."""

    __slots__ = ["hubbard", "lhubbard", "average", "gexp", "shell_resolved"]

    class Cache(Interaction.Cache, TensorLike):
        """
        Cache for Coulomb matrix in ES2.
        """

        __store: Store | None
        """Storage for cache (required for culling)."""

        mat: Tensor
        """Coulomb matrix."""

        shell_resolved: bool
        """Electrostatics is shell-resolved (default: `True`)."""

        __slots__ = ["__store", "mat", "shell_resolved"]

        def __init__(
            self,
            mat: Tensor,
            shell_resolved: bool = True,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> None:
            super().__init__(
                device=device if device is None else mat.device,
                dtype=dtype if dtype is None else mat.dtype,
            )
            self.mat = mat
            self.shell_resolved = shell_resolved
            self.__store = None

        class Store:
            """
            Storage container for cache containing `__slots__` before culling.
            """

            mat: Tensor
            """Coulomb matrix"""

            def __init__(self, mat: Tensor) -> None:
                self.mat = mat

        def cull(self, conv: Tensor, slicers: Slicers) -> None:
            if self.__store is None:
                self.__store = self.Store(self.mat)

            slicer = slicers["shell"] if self.shell_resolved else slicers["atom"]
            self.mat = self.mat[[~conv, *slicer, *slicer]]

        def restore(self) -> None:
            if self.__store is None:
                raise RuntimeError("Nothing to restore. Store is empty.")

            self.mat = self.__store.mat

    def __init__(
        self,
        hubbard: Tensor,
        lhubbard: Tensor | None = None,
        average: AveragingFunction = harmonic_average,
        gexp: Tensor = torch.tensor(xtb.DEFAULT_ES2_GEXP),
        shell_resolved: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)

        self.hubbard = hubbard.to(self.device).type(self.dtype)
        self.lhubbard = (
            lhubbard if lhubbard is None else lhubbard.to(self.device).type(self.dtype)
        )
        self.gexp = gexp.to(self.device).type(self.dtype)
        self.average = average

        self.shell_resolved = shell_resolved and lhubbard is not None

    def get_cache(
        self,
        numbers: Tensor,
        positions: Tensor,
        ihelp: IndexHelper,
    ) -> ES2.Cache:
        """
        Obtain the cache object containing the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        ES2.Cache
            Cache object for second order electrostatics.

        Note
        ----
        The cache of an interaction requires `positions` as they do not change
        during the self-consistent charge iterations.
        """
        return self.Cache(
            self.get_shell_coulomb_matrix(numbers, positions, ihelp)
            if self.shell_resolved
            else self.get_atom_coulomb_matrix(numbers, positions, ihelp),
            shell_resolved=self.shell_resolved,
        )

    def get_atom_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """
        Calculate the atom-resolved Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """
        mask = real_pairs(numbers, diagonal=True)

        mat = CoulombMatrixAG.apply(
            mask, positions, ihelp, self.hubbard, self.gexp, self.average
        )
        assert mat is not None

        return mat

    def get_shell_coulomb_matrix(
        self, numbers: Tensor, positions: Tensor, ihelp: IndexHelper
    ) -> Tensor:
        """
        Calculate the Coulomb matrix.

        Parameters
        ----------
        numbers : Tensor
            Atomic numbers of all atoms in the system.
        positions : Tensor
            Cartesian coordinates of all atoms in the system.
        ihelp : IndexHelper
            Index mapping for the basis set.

        Returns
        -------
        Tensor
            Coulomb matrix.
        """
        if self.lhubbard is None:
            raise ValueError("No 'lhubbard' parameters set.")

        lh = ihelp.spread_ushell_to_shell(self.lhubbard)
        h = lh * ihelp.spread_uspecies_to_shell(self.hubbard)

        # masks
        mask = real_pairs(numbers, diagonal=True)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        dist_gexp = ihelp.spread_atom_to_shell(
            torch.where(
                mask,
                torch.pow(
                    torch.cdist(
                        positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
                    ),
                    self.gexp,
                ),
                positions.new_tensor(torch.finfo(positions.dtype).eps),
            ),
            (-1, -2),
        )

        # Eq.30: averaging function for hardnesses (Hubbard parameter)
        avg = self.average(h)

        # Eq.26: Coulomb matrix
        return 1.0 / torch.pow(dist_gexp + torch.pow(avg, -self.gexp), 1.0 / self.gexp)

    def get_atom_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        return (
            torch.zeros_like(charges)
            if self.shell_resolved
            else 0.5 * charges * self.get_atom_potential(charges, cache)
        )

    def get_shell_energy(self, charges: Tensor, cache: Cache) -> Tensor:
        return (
            0.5 * charges * self.get_shell_potential(charges, cache)
            if self.shell_resolved
            else torch.zeros_like(charges)
        )

    def get_atom_potential(self, charges: Tensor, cache: Cache) -> Tensor:
        return (
            torch.zeros_like(charges)
            if self.shell_resolved
            else torch.einsum("...ik,...k->...i", cache.mat, charges)
        )

    def get_shell_potential(self, charges: Tensor, cache: ES2.Cache) -> Tensor:
        """
        Calculate shell-resolved potential. Zero if this interaction is only
        atom-resolved.

        Parameters
        ----------
        charges : Tensor
            Shell-resolved partial charges.
        cache : ES2.Cache
            Cache object for second order electrostatics.

        Returns
        -------
        Tensor
            Shell-resolved potential.
        """
        return (
            torch.einsum("...ik,...k->...i", cache.mat, charges)
            if self.shell_resolved
            else torch.zeros_like(charges)
        )

    def get_atom_gradient(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        cache: ES2.Cache,
    ) -> Tensor:
        if self.shell_resolved:
            return torch.zeros_like(positions)

        mask = real_pairs(numbers, diagonal=True)

        distances = torch.where(
            mask,
            torch.cdist(
                positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
            ),
            positions.new_tensor(0.0),
        )

        # (n_batch, atoms_i, atoms_j, 3)
        rij = torch.where(
            mask.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            positions.new_tensor(0.0),
        )

        # (n_batch, atoms_i) -> (n_batch, atoms_i, 1)
        charges = charges.unsqueeze(-1)

        # (n_batch, atoms_i, atoms_j) * (n_batch, atoms_i, 1)
        # every column is multiplied by the charge vector
        dmat = (
            -(distances ** (self.gexp - 2.0)) * cache.mat * cache.mat**self.gexp
        ) * charges

        # (n_batch, atoms_i, atoms_j) -> (n_batch, atoms_i, atoms_j, 3)
        dmat = dmat.unsqueeze(-1) * rij

        # (n_batch, atoms_i, atoms_j, 3) -> (n_batch, atoms_i, 3)
        return torch.einsum("...ijx,...jx->...ix", dmat, charges)

    def get_shell_gradient(
        self,
        numbers: Tensor,
        positions: Tensor,
        charges: Tensor,
        cache: ES2.Cache,
        ihelp: IndexHelper,
    ) -> Tensor:
        if not self.shell_resolved:
            return torch.zeros_like(positions)

        mask = real_pairs(numbers, diagonal=True)

        # all distances to the power of "gexp" (R^2_AB from Eq.26)
        distances = ihelp.spread_atom_to_shell(
            torch.where(
                mask,
                torch.cdist(
                    positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"
                ),
                positions.new_tensor(torch.finfo(positions.dtype).eps),
            ),
            (-1, -2),
        )

        # (n_batch, shells_i, shells_j, 3)
        positions = batch.index(positions, ihelp.shells_to_atom)
        mask = ihelp.spread_atom_to_shell(mask, (-2, -1))
        rij = torch.where(
            mask.unsqueeze(-1),
            positions.unsqueeze(-2) - positions.unsqueeze(-3),
            positions.new_tensor(0.0),
        )

        # (n_batch, shells_i) -> (n_batch, shells_i, 1)
        charges = charges.unsqueeze(-1)

        # (n_batch, shells_i, shells_j) * (n_batch, shells_i, 1)
        # every column is multiplied by the charge vector
        dmat = (
            -(distances ** (self.gexp - 2.0)) * cache.mat * cache.mat**self.gexp
        ) * charges

        # (n_batch, shells_i, shells_j) -> (n_batch, shells_i, shells_j, 3)
        dmat = dmat.unsqueeze(-1) * rij

        # (n_batch, shells_i, shells_j, 3) -> (n_batch, atoms, shells_j, 3)
        dmat = ihelp.reduce_shell_to_atom(dmat, dim=-3, extra=True)

        # (n_batch, atoms, shells_j, 3) -> (n_batch, atoms, 3)
        return torch.einsum("...ijx,...jx->...ix", dmat, charges)

    def _get_atom_gradient(self, energy: Tensor, positions: Tensor) -> Tensor:
        """
        Calculates nuclear gradient of an second order electrostatic energy
        contribution via PyTorch's autograd engine.

        Parameters
        ----------
        energy : Tensor
            Energy that will be differentiated.
        positions : Tensor
            Nuclear positions. Needs `requires_grad=True`.

        Returns
        -------
        Tensor
            Nuclear gradient of `energy`.

        Raises
        ------
        RuntimeError
            `positions` tensor does not have `requires_grad=True`.
        """
        if positions.requires_grad is False:
            raise RuntimeError("Position tensor needs `requires_grad=True`.")

        # avoid autograd call if energy is zero (autograd fails anyway)
        if torch.equal(energy, torch.zeros_like(energy)):
            return torch.zeros_like(positions)

        (gradient,) = torch.autograd.grad(
            energy, positions, grad_outputs=torch.ones_like(energy)
        )
        return gradient


def coulomb_matrix_atom(
    mask: Tensor,
    positions: Tensor,
    ihelp: IndexHelper,
    hubbard: Tensor,
    gexp: Tensor,
    average: AveragingFunction,
) -> Tensor:
    """
    Calculate the atom-resolved Coulomb matrix.

    Parameters
    ----------
    mask : Tensor
        Mask from atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    ihelp : IndexHelper
        Index mapping for the basis set.
    hubbard : Tensor
        Hubbard parameters of all elements.
    gexp: Tensor
        Exponent of the second-order Coulomb interaction (default: 2.0).
    average: AveragingFunction
        Function to use for averaging the Hubbard parameters (default:
        `~dxtb.coulomb.average.harmonic_average`).

    Returns
    -------
    Tensor
        Coulomb matrix.
    """
    h = ihelp.spread_uspecies_to_atom(hubbard)

    dist = torch.cdist(
        positions,
        positions,
        p=2,
        compute_mode="use_mm_for_euclid_dist",
    )

    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)
    zero = positions.new_tensor(0.0)

    # all distances to the power of "gexp" (R^2_AB from Eq.26)
    dist_gexp = torch.where(
        mask,
        # eps to avoid nan in double backward (negative base?)
        torch.pow(dist + eps, gexp),
        eps,
    )

    # Eq.30: averaging function for hardnesses (Hubbard parameter)
    avg = torch.where(mask, average(h), zero)

    # Eq.26: Coulomb matrix
    tmp = torch.where(mask, torch.pow(avg + eps, -gexp), zero)
    return torch.where(mask, 1.0 / torch.pow(dist_gexp + tmp, 1.0 / gexp), zero)


def coulomb_matrix_atom_gradient(
    mask: Tensor, positions: Tensor, mat: Tensor, gexp: Tensor
) -> Tensor:
    """
    Nuclear gradient of atom-resolved Coulomb matrix.

    Parameters
    ----------
    mask : Tensor
        Mask from atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    mat : Tensor
        Atom-resolved Coulomb matrix.
    gexp: Tensor
        Exponent of the second-order Coulomb interaction (default: 2.0).

    Returns
    -------
    Tensor
        Derivative of atom-resolved Coulomb matrix. The derivative has the
        following shape: `(n_batch, atoms_i, atoms_j, 3)`.
    """
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        positions.new_tensor(0.0),
    )

    # (n_batch, atoms_i, atoms_j, 3)
    rij = torch.where(
        mask.unsqueeze(-1),
        positions.unsqueeze(-2) - positions.unsqueeze(-3),
        positions.new_tensor(0.0),
    )

    # (n_batch, atoms_i, atoms_j)
    dmat = -(distances ** (gexp - 2.0)) * mat * mat**gexp

    # (n_batch, atoms_i, atoms_j) -> (n_batch, atoms_i, atoms_j, 3)
    return dmat.unsqueeze(-1) * rij


class CoulombMatrixAG(torch.autograd.Function):
    """
    Autograd function for Coulomb matrix.
    """

    @staticmethod
    def forward(
        ctx,
        mask: Tensor,
        positions: Tensor,
        ihelp: IndexHelper,
        hubbard: Tensor,
        gexp: Tensor,
        average: AveragingFunction,
    ) -> Tensor:
        with torch.enable_grad():
            mat = coulomb_matrix_atom(mask, positions, ihelp, hubbard, gexp, average)

        ctx.save_for_backward(mat, positions, mask, gexp, hubbard)

        return mat.clone()

    @staticmethod
    def backward(
        ctx, grad_out: Tensor
    ) -> tuple[
        None,  # mask
        None | Tensor,  # positions
        None,  # ihelp
        None | Tensor,  # hubbard
        None | Tensor,  # gexp
        None,  # average
    ]:
        # initialize gradients with `None`
        positions_bar = hubbard_bar = gexp_bar = None

        # check which of the input variables of `forward()` requires gradients
        _, grad_positions, _, grad_hubbard, grad_gexp, _ = ctx.needs_input_grad

        mat, positions, mask, gexp, hubbard = ctx.saved_tensors

        # analytical gradient for positions
        if grad_positions:
            # (n_batch, n_atoms, n_atoms, 3)
            g = coulomb_matrix_atom_gradient(mask, positions, mat, gexp)

            # vjp: (nb, na, na) * (nb, na, na, 3) -> (nb, na, 3)
            _gi = torch.einsum("...ij,...ijd->...id", grad_out, g)
            _gj = torch.einsum("...ij,...ijd->...jd", grad_out, g)
            positions_bar = _gi - _gj

        # automatic gradient for parameters
        if grad_hubbard:
            (hubbard_bar,) = torch.autograd.grad(
                mat,
                hubbard,
                grad_outputs=grad_out,
                create_graph=True,
            )

        if grad_gexp:
            (gexp_bar,) = torch.autograd.grad(
                mat,
                gexp,
                grad_outputs=grad_out,
                create_graph=True,
            )

        return None, positions_bar, None, hubbard_bar, gexp_bar, None


def new_es2(
    numbers: Tensor,
    par: Param,
    shell_resolved: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> ES2 | None:
    """
    Create new instance of ES2.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms.
    par : Param
        Representation of an extended tight-binding model.
    shell_resolved: bool
        Electrostatics is shell-resolved.

    Returns
    -------
    ES2 | None
        Instance of the ES2 class or `None` if no ES2 is used.
    """

    if hasattr(par, "charge") is False or par.charge is None:
        return None

    unique = torch.unique(numbers)
    hubbard = get_elem_param(unique, par.element, "gam")
    lhubbard = (
        get_elem_param(unique, par.element, "lgam") if shell_resolved is True else None
    )
    average = averaging_function[par.charge.effective.average]
    gexp = torch.tensor(par.charge.effective.gexp)

    return ES2(hubbard, lhubbard, average, gexp, device=device, dtype=dtype)
