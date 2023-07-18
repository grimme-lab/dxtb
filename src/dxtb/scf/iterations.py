from __future__ import annotations

import torch

from .._types import Tensor
from .conversions import potential_to_charges, charges_to_potential

"""
Iterations of physical properties implemented as pure functions 
in order to avoid RAM leak due to circular references.
"""


def iterate_potential(potential: Tensor, data, cfg, interactions) -> Tensor:
    charges = potential_to_charges(potential, data, cfg)
    return charges_to_potential(charges, interactions, data)


# TODO
# def iterate_charges(charges: Tensor) -> Tensor:
#     """
#     Perform single self-consistent iteration.

#     Parameters
#     ----------
#     charges : Tensor
#         Orbital-resolved partial charges vector.

#     Returns
#     -------
#     Tensor
#         New orbital-resolved partial charges vector.
#     """
#     if self.scf_options.get("verbosity", defaults.VERBOSITY) > 0:
#         if charges.ndim < 2:  # pragma: no cover
#             energy = self.get_energy(charges).sum(-1).detach().clone()
#             ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

#             density = self._data.density.detach().clone()
#             pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

#             q = charges.detach().clone()
#             qdiff = torch.linalg.vector_norm(self._data.old_charges - q)

#             print(
#                 f"{self._data.iter:3}   {energy: .16E}  {ediff: .6E} "
#                 f"{pnorm: .6E}   {qdiff: .6E}"
#             )

#             self._data.old_energy = energy
#             self._data.old_charges = q
#             self._data.old_density = density
#             self._data.iter += 1
#         else:
#             energy = self.get_energy(charges).detach().clone()
#             ediff = torch.linalg.norm(self._data.old_energy - energy)

#             density = self._data.density.detach().clone()
#             pnorm = torch.linalg.norm(self._data.old_density - density)

#             q = charges.detach().clone()
#             qdiff = torch.linalg.norm(self._data.old_charges - q)

#             print(
#                 f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
#                 f"{qdiff: .6E}"
#             )

#             self._data.old_energy = energy
#             self._data.old_charges = q
#             self._data.old_density = density
#             self._data.iter += 1

#     if self.fwd_options["verbose"] > 1:  # pragma: no cover
#         print(f"energy: {self.get_energy(charges).sum(-1)}")
#     potential = self.charges_to_potential(charges)
#     return self.potential_to_charges(potential)

# def iterate_potential(potential: Tensor) -> Tensor:
#     """
#     Perform single self-consistent iteration.

#     Parameters
#     ----------
#     potential: Tensor
#         Potential vector for each orbital partial charge.

#     Returns
#     -------
#     Tensor
#         New potential vector for each orbital partial charge.
#     """

#     charges = self.potential_to_charges(potential)
#     if self.scf_options["verbosity"] > 0:
#         if charges.ndim < 2:  # pragma: no cover
#             energy = self.get_energy(charges).sum(-1).detach().clone()
#             ediff = torch.linalg.vector_norm(self._data.old_energy - energy)

#             density = self._data.density.detach().clone()
#             pnorm = torch.linalg.matrix_norm(self._data.old_density - density)

#             q = charges.detach().clone()
#             qdiff = torch.linalg.vector_norm(self._data.old_charges - q)

#             print(
#                 f"{self._data.iter:3}   {energy: .16E}  {ediff: .6E} "
#                 f"{pnorm: .6E}   {qdiff: .6E}"
#             )

#             self._data.old_energy = energy
#             self._data.old_charges = q
#             self._data.old_density = density
#             self._data.iter += 1
#         else:
#             energy = self.get_energy(charges).detach().clone()
#             ediff = torch.linalg.norm(self._data.old_energy - energy)

#             density = self._data.density.detach().clone()
#             pnorm = torch.linalg.norm(self._data.old_density - density)

#             q = charges.detach().clone()
#             qdiff = torch.linalg.norm(self._data.old_charges - q)

#             print(
#                 f"{self._data.iter:3}   {energy.sum(): .16E}  {ediff: .6E} "
#                 f"{qdiff: .6E}"
#             )

#             self._data.old_energy = energy
#             self._data.old_charges = q
#             self._data.old_density = density
#             self._data.iter += 1

#     return self.charges_to_potential(charges)

# def iterate_fockian(self, fockian: Tensor) -> Tensor:
#     """
#     Perform single self-consistent iteration using the Fock matrix.

#     Parameters
#     ----------
#     fockian : Tensor
#         Fock matrix.

#     Returns
#     -------
#     Tensor
#         New Fock matrix.
#     """
#     self._data.density = self.hamiltonian_to_density(fockian)
#     charges = self.density_to_charges(self._data.density)
#     potential = self.charges_to_potential(charges)
#     self._data.hamiltonian = self.potential_to_hamiltonian(potential)

#     return self._data.hamiltonian

iter_options = {"potential": iterate_potential}
# iter_options = {"charge": iterate_charges, "charges": iterate_charges,"potential": iterate_potential,"fock": iterate_fockian,}
"""Possible physical values to be iterated during SCF procedure."""
