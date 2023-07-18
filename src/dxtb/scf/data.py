import torch

from .._types import Slicers, Tensor
from ..basis import IndexHelper
from ..interaction import InteractionList


class _Data:
    """
    Restart data for the singlepoint calculation.
    """

    numbers: Tensor
    """Atomic numbers"""

    hcore: Tensor
    """Core Hamiltonian"""

    overlap: Tensor
    """Overlap matrix"""

    occupation: Tensor
    """Occupation numbers (shape: [..., 2, orbs])"""

    n0: Tensor
    """Reference occupations for each orbital (shape: [..., orbs])"""

    ihelp: IndexHelper
    """Index mapping for the basis set"""

    cache: InteractionList.Cache
    """Restart data for the interactions"""

    energy: Tensor
    """Electronic energy (shape: [..., orbs])"""

    hamiltonian: Tensor
    """Self-consistent Hamiltonian (shape: [..., orbs, orbs])"""

    density: Tensor
    """Density matrix"""

    evals: Tensor
    """
    Orbital energies, i.e., eigenvalues of Fock matrix
    (shape: [..., orbs])
    """

    evecs: Tensor
    """
    LCAO coefficients, i.e., eigenvectors of Fock matrix
    (shape: [..., orbs, orbs])
    """

    iter: int
    """Number of iterations."""

    def __init__(
        self,
        hcore: Tensor,
        overlap: Tensor,
        occupation: Tensor,
        n0: Tensor,
        numbers: Tensor,
        ihelp: IndexHelper,
        cache: InteractionList.Cache,
    ) -> None:
        self.hcore = hcore
        self.overlap = overlap
        self.occupation = occupation
        self.n0 = n0
        self.numbers = numbers
        self.ihelp = ihelp
        self.cache = cache
        self.init_zeros()

        self.iter = 1

    def init_zeros(self) -> None:
        self.energy = torch.zeros_like(self.n0)
        self.hamiltonian = torch.zeros_like(self.hcore)
        self.density = torch.zeros_like(self.hcore)
        self.evals = torch.zeros_like(self.n0)
        self.evecs = torch.zeros_like(self.hcore)

        self.old_charges = torch.zeros_like(self.energy)
        self.old_energy = torch.zeros_like(self.numbers)
        self.old_density = torch.zeros_like(self.density)

    def reset(self) -> None:
        self.iter = 0
        self.init_zeros()

    def cull(self, conv: Tensor, slicers: Slicers) -> None:
        onedim = [~conv, *slicers["orbital"]]
        twodim = [~conv, *slicers["orbital"], *slicers["orbital"]]

        self.numbers = self.numbers[[~conv, *slicers["atom"]]]
        self.overlap = self.overlap[twodim]
        self.hamiltonian = self.hamiltonian[twodim]
        self.hcore = self.hcore[twodim]
        self.occupation = self.occupation[twodim]
        self.evecs = self.evecs[twodim]
        self.evals = self.evals[onedim]
        self.energy = self.energy[onedim]
        self.n0 = self.n0[onedim]
        self.ihelp.cull(conv, slicers=slicers)
        self.cache.cull(conv, slicers=slicers)

        self.old_charges = self.old_charges[onedim]
        self.old_energy = self.old_energy[onedim]
        self.old_density = self.old_density[twodim]

    def clean(self) -> tuple[Tensor]:
        """
        Detach and return tensor attributes linked to object. The following
        attributes are returned: density, hamiltonian, energy, evals, and evecs.

        This function is used to free up memory by breaking the pytorch autograd
        graph-loop linked to this object. This allows to free memory and avoid RAM leaks.s

        Returns:
            tuple[Tensor]: A tuple containing the attributes: density, hamiltonian, energy, evals, and evecs.

        Raises:
            AttributeError: If any of the attributes: density, hamiltonian, energy, evals, or evecs
            does not exist in the instance.
        """
        density = self.density
        hamiltonian = self.hamiltonian
        energy = self.energy
        evals = self.evals
        evecs = self.evecs
        del self.density
        del self.hamiltonian
        del self.energy
        del self.evals
        del self.evecs
        return density, hamiltonian, energy, evals, evecs
