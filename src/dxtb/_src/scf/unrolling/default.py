# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SCF Unrolling: Standard Variant
===============================

Standard implementation of SCF iterations with unrolling or full gradient
tracking for the backward, i.e., no special treatment of the backward pass.

In batched calculations, the SCF is performed for each system in the batch
separately. Converged systems are removed from the batch, and the SCF is
continued until all systems are converged.
"""

from __future__ import annotations

from math import prod

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import (
    Charges,
    ContainerLayout,
    Potential,
)
from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any, Literal, Slicers, Tensor, exceptions, overload
from dxtb._src.utils import t2int

from .base import BaseTSCF

__all__ = ["SelfConsistentFieldFull"]


class SelfConsistentFieldFull(BaseTSCF):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    This SCF class uses a straightfoward implementation of simple or Anderson
    mixing (taken from TBMaLT). Therefore, the gradient tracking is enabled
    for all iterations.

    To remedy some cost and avoid overconvergence in batched calculations,
    converged systems are removed (culled) from the batch.
    """

    @overload
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[True] = True,
    ) -> Charges: ...

    @overload
    def scf(
        self,
        guess: Tensor,
        return_charges: Literal[False] = False,
    ) -> Charges | Potential | Tensor: ...

    def scf(
        self, guess: Tensor, return_charges: bool = True
    ) -> Charges | Potential | Tensor:
        # "SCF function"
        fcn = self._fcn

        maxiter = self.config.maxiter
        batched = self.config.batch_mode

        # Evaluate initial guess outside of SCF loop to make maxiter=0 possible.
        q_new = fcn(guess)
        if OutputHandler.verbosity >= 3:  # pragma: no cover
            charges = self.converged_to_charges(q_new)
            energy = self.get_energy(charges)
            self._print(charges, energy)

        if maxiter == 0:
            if return_charges is True:
                return self.converged_to_charges(q_new)
            return q_new

        # Also mix the initial guess with the first SCF iteration
        if self.config.mix_guess is True:
            q = self.mixer.iter(q_new, guess)
        else:
            q = q_new

        # single-system (non-batched) case, which does not require culling
        if batched == 0:
            for _ in range(maxiter):
                q_new = fcn(q)

                # Important: Calculate energy with charges before mixing!
                if OutputHandler.verbosity >= 3:  # pragma: no cover
                    charges = self.converged_to_charges(q_new)
                    energy = self.get_energy(charges)
                    self._print(charges, energy)

                q = self.mixer.iter(q_new, q)

                if self.mixer.converged:
                    # Do not return mixed charges here, but from last SCF call!
                    q_converged = q_new
                    break

                if self.config.damp_dynamic is True:
                    # Switch off damping if the norm of the difference is small
                    if self.mixer.delta_norm < 0.1:
                        self.mixer.options["damp"] = (
                            self.config.damp_dynamic_factor
                        )

            else:
                msg = (
                    f"SCF does not converge after {maxiter} cycles using "
                    f"{self.mixer.label} mixing with a damping factor of "
                    f"{self.mixer.options['damp']}."
                )
                if self.config.force_convergence is True:
                    raise exceptions.SCFConvergenceError(msg)

                # only issue warning, return anyway
                OutputHandler.warn(msg, exceptions.SCFConvergenceWarning)

                # Do not return mixed charges here, but from last SCF call!
                q_converged = q_new

            if return_charges is True:
                return self.converged_to_charges(q_converged)
            return q_converged

        # batched SCF with culling
        culled = True
        ch = torch.zeros_like(self._data.hamiltonian)
        cevals = torch.zeros_like(self._data.evals)
        cevecs = torch.zeros_like(self._data.evecs)
        ce = torch.zeros_like(self._data.energy)
        co = torch.zeros_like(self._data.occupation)
        cd = torch.zeros_like(self._data.density)
        n0 = self._data.n0
        nel = self._data.nel
        numbers = self._data.numbers
        charges_data = self._data.charges.copy()
        potential_data = self._data.potential.copy()
        q_converged = torch.full_like(guess, defaults.PADNZ, device=self.device)

        overlap = self._data.ints.overlap
        hcore = self._data.ints.hcore
        dipole = self._data.ints.dipole
        quad = self._data.ints.quadrupole

        idxs = torch.arange(guess.size(0), device=self.device)
        converged = torch.full(idxs.shape, False, device=self.device)
        wave_norb = self._data.ihelp.nao
        nsh = self._data.ihelp.nsh
        nat = self._data.ihelp.nat
        container_target = self.config.scp_mode != labels.SCP_MODE_FOCK
        nprop = q.shape[1] if container_target else 0

        def matrix_index(index: Tensor, size: int) -> tuple[Any, ...]:
            return (index, ..., slice(0, size), slice(0, size))

        def vector_index(index: Tensor, size: int) -> tuple[Any, ...]:
            return (index, ..., slice(0, size))

        def update_layout(
            layout: ContainerLayout, batch: int, norb: int, natom: int
        ) -> None:
            spin_shape = (layout.nspin,) if layout.nspin == 2 else ()
            if layout.mono is not None:
                layout.mono = torch.Size((batch, *spin_shape, norb))
            if layout.dipole is not None:
                layout.dipole = torch.Size(
                    (batch, *spin_shape, natom, defaults.DP_SHAPE)
                )
            if layout.quad is not None:
                layout.quad = torch.Size(
                    (batch, *spin_shape, natom, defaults.QP_SHAPE)
                )

        def layout_extent(layout: ContainerLayout) -> int:
            return max(prod(shape[1:]) for shape in layout.component_shapes)

        for _ in range(maxiter):
            q_new = fcn(q)

            if OutputHandler.verbosity >= 3:  # pragma: no cover
                charges = self.converged_to_charges(q_new)
                energy = self.get_energy(charges)
                self._print(charges, energy)

            q = self.mixer.iter(q_new, q)
            conv = self.mixer.converged
            if not conv.any():
                continue

            # If the original batch converges simultaneously, no state was
            # culled and the current tensors are already complete.
            if conv.all() and idxs.numel() == guess.shape[0]:
                q_converged = q_new
                converged[:] = True
                culled = False
                break

            iconv = idxs[conv]
            if container_target:
                # Rows are orbital-/atom-major (see `Container.as_tensor`),
                # hence the converged prefix can be stored directly.
                q_converged[iconv, :nprop, : q_new.shape[-1]] = q_new[conv]
            else:
                q_converged[matrix_index(iconv, wave_norb)] = q_new[
                    matrix_index(conv, wave_norb)
                ]

            mat_out = matrix_index(iconv, wave_norb)
            mat_in = matrix_index(conv, wave_norb)
            vec_out = vector_index(iconv, wave_norb)
            vec_in = vector_index(conv, wave_norb)
            ch[mat_out] = self._data.hamiltonian[mat_in]
            cevecs[mat_out] = self._data.evecs[mat_in]
            cevals[vec_out] = self._data.evals[vec_in]
            ce[iconv, :wave_norb] = self._data.energy[conv, :wave_norb]
            co[vec_out] = self._data.occupation[vec_in]
            cd[mat_out] = self._data.density[mat_in]
            converged[iconv] = True

            if conv.all():
                break

            wave_norb_new = t2int(
                self._data.ihelp.orbitals_per_shell[~conv, ...].sum(-1).max()
            )
            nsh_new = t2int(
                self._data.ihelp.shells_per_atom[~conv, ...].sum(-1).max()
            )
            nat_new = t2int(
                self._data.numbers[~conv, ...].count_nonzero(dim=-1).max()
            )

            slicers: Slicers = {
                "orbital": (...,),
                "shell": (...,),
                "atom": (...,),
            }
            if wave_norb_new < wave_norb:
                slicers["orbital"] = [slice(0, wave_norb_new)]
            if nsh_new < nsh:
                slicers["shell"] = [slice(0, nsh_new)]
            if nat_new < nat:
                slicers["atom"] = [slice(0, nat_new)]

            self._data.cull(conv, slicers=slicers)
            idxs = idxs[~conv]
            wave_norb, nsh, nat = wave_norb_new, nsh_new, nat_new
            update_layout(self._data.charges, len(idxs), wave_norb, nat)
            update_layout(self._data.potential, len(idxs), wave_norb, nat)

            if container_target:
                layout = (
                    self._data.charges
                    if self.config.scp_mode == labels.SCP_MODE_CHARGE
                    else self._data.potential
                )
                q_extent = layout_extent(layout)
                q = q[~conv, :nprop, :q_extent]
                q_new = q_new[~conv, :nprop, :q_extent]
                mixer_slices = [slice(0, nprop), slice(0, q_extent)]
            else:
                qindex = matrix_index(~conv, wave_norb)
                q = q[qindex]
                q_new = q_new[qindex]
                spin_slices = [slice(0, 2)] if self.nspin == 2 else []
                mixer_slices = [
                    *spin_slices,
                    slice(0, wave_norb),
                    slice(0, wave_norb),
                ]

            self.mixer.cull(conv, slicers=mixer_slices)

        else:
            msg = (
                f"SCF does not converge after '{maxiter}' cycles using "
                f"'{self.mixer.label}' mixing with a damping factor of "
                f"'{self.mixer.options['damp']}'."
            )
            if self.config.force_convergence is True:
                raise exceptions.SCFConvergenceError(msg)

            if container_target:
                q_converged[idxs, :nprop, : q_new.shape[-1]] = q_new
            else:
                q_converged[matrix_index(idxs, wave_norb)] = q_new[
                    matrix_index(
                        torch.ones(
                            len(idxs), dtype=torch.bool, device=self.device
                        ),
                        wave_norb,
                    )
                ]

            if (~converged).all():
                culled = False

            all_idxs = torch.arange(guess.size(0), device=self.device)
            msg_converged = (
                "\nForced convergence is turned off. The calculation will "
                "continue with the current unconverged charges."
                f"\nIn total, {len(idxs)} systems did not converge "
                f"({idxs.tolist()}), and {int(converged.count_nonzero())} "
                f"converged ({all_idxs[converged].tolist()})."
            )
            OutputHandler.warn(
                msg + msg_converged, exceptions.SCFConvergenceWarning
            )

        if culled is True:
            if not converged.all():
                mat_out = matrix_index(idxs, wave_norb)
                vec_out = vector_index(idxs, wave_norb)
                ch[mat_out] = self._data.hamiltonian
                cevecs[mat_out] = self._data.evecs
                cevals[vec_out] = self._data.evals
                ce[idxs, :wave_norb] = self._data.energy
                co[vec_out] = self._data.occupation
                cd[mat_out] = self._data.density

            self._data.evals = cevals
            self._data.evecs = cevecs
            self._data.energy = ce
            self._data.hamiltonian = ch
            self._data.occupation = co
            self._data.density = cd
            self._data.charges = charges_data
            self._data.potential = potential_data
            self._data.n0 = n0
            self._data.nel = nel
            self._data.numbers = numbers

            self._data.ints.run_checks = False
            self._data.ints.overlap = overlap
            self._data.ints.hcore = hcore
            if self._data.ints.dipole is not None and dipole is not None:
                self._data.ints.dipole = dipole
            if self._data.ints.quadrupole is not None and quad is not None:
                self._data.ints.quadrupole = quad
            self._data.ints.run_checks = True
            self._data.ihelp.restore()
            self._data.cache.restore()

        if return_charges is True:
            return self.converged_to_charges(q_converged)
        return q_converged
