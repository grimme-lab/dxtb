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

import torch

from dxtb import OutputHandler
from dxtb._src.components.interactions import Charges, Potential
from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Literal, Slicers, Tensor, exceptions, overload
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

        q = guess

        # single-system (non-batched) case, which does not require culling
        if batched == 0:
            for _ in range(maxiter):
                q_new = fcn(q)
                q = self.mixer.iter(q_new, q)

                if self.mixer.converged:
                    q_converged = q
                    break

            else:
                msg = (
                    f"\nSCF does not converge after {maxiter} cycles using "
                    f"{self.mixer.label} mixing with a damping factor of "
                    f"{self.mixer.options['damp']}."
                )
                if self.config.force_convergence is True:
                    raise exceptions.SCFConvergenceError(msg)

                # only issue warning, return anyway
                OutputHandler.warn(msg, exceptions.SCFConvergenceWarning)
                q_converged = q

            if return_charges is True:
                return self.converged_to_charges(q_converged)
            return q_converged

        # batched SCF with culling
        culled = True

        # Initialize variables that change throughout the SCF. Later, we
        # fill these with the converged values and simultaneously cull
        # them from `self._data`
        ch = torch.zeros_like(self._data.hamiltonian)
        cevals = torch.zeros_like(self._data.evals)
        cevecs = torch.zeros_like(self._data.evecs)
        ce = torch.zeros_like(self._data.evals)
        co = torch.zeros_like(self._data.occupation)
        cd = torch.zeros_like(self._data.density)
        n0 = self._data.n0
        numbers = self._data.numbers
        charges_data = self._data.charges.copy()
        potential_data = self._data.potential.copy()

        # shape: (nb, <number of moments>, norb)
        q_converged = torch.full_like(guess, defaults.PADNZ)

        overlap = self._data.ints.overlap
        hcore = self._data.ints.hcore
        dipole = self._data.ints.dipole
        quad = self._data.ints.quadrupole

        # indices for systems in batch, required for culling
        idxs = torch.arange(guess.size(0))

        # tracker for converged systems
        converged = torch.full(idxs.shape, False)

        # maximum number of orbitals in batch
        norb = self._data.ihelp.nao
        _norb = self._data.ihelp.nao
        nsh = self._data.ihelp.nsh
        nat = self._data.ihelp.nat

        # Here, we account for cases, in which the number of
        # orbitals is smaller than the number of atoms times 3 (6)
        # after culling. We specifically avoid culling, as this
        # would severly mess up the shapes involved.
        if q.shape[1] == 2:
            norb = max(norb, nat * defaults.DP_SHAPE)
        elif q.shape[1] == 3:
            norb = max(norb, nat * defaults.QP_SHAPE)

        # We need to specify the number of multipole dimensions for the
        # culling to work properly later. If we are converging the Fock
        # matrix, there is no such thing as multipole dimensions. However,
        # we will shamelessly use this as the second dimension of the Fock
        # matrix even modify it in for the culling process.
        mpdim = q.shape[1]

        # initialize slicers for culling
        slicers: Slicers = {
            "orbital": (...,),
            "shell": (...,),
            "atom": (...,),
        }

        for _ in range(maxiter):
            q_new = fcn(q)
            q = self.mixer.iter(q_new, q)

            conv = self.mixer.converged
            if conv.any():
                # Simultaneous convergence does not require culling.
                # Occurs if batch size equals amount of True in `conv`.
                if guess.shape[0] == conv.count_nonzero():
                    q_converged = q
                    converged[:] = True
                    culled = False
                    break

                # save all necessary variables for converged system
                iconv = idxs[conv]
                q_converged[iconv, :mpdim, :norb] = q[conv, ..., :]
                ch[iconv, :norb, :norb] = self._data.hamiltonian[conv, :, :]
                cevecs[iconv, :norb, :norb] = self._data.evecs[conv, :, :]
                cevals[iconv, :norb] = self._data.evals[conv, :]
                ce[iconv, :norb] = self._data.energy[conv, :]
                co[iconv, :norb, :norb] = self._data.occupation[conv, :, :]
                cd[iconv, :norb, :norb] = self._data.density[conv, :, :]

                # update convergence tracker
                converged[iconv] = True

                # end SCF if all systems are converged
                if conv.all():
                    break

                # cull `orbitals_per_shell` (`shells_per_atom`) to
                # calculate maximum number of orbitals (shells), which
                # corresponds to the maximum padding
                norb_new = self._data.ihelp.orbitals_per_shell[~conv, ...].sum(-1).max()
                _norb_new = norb_new
                nsh_new = self._data.ihelp.shells_per_atom[~conv, ...].sum(-1).max()
                nat_new = self._data.numbers[~conv, ...].count_nonzero(dim=-1).max()

                # Here, we account for cases, in which the number of
                # orbitals is smaller than the number of atoms times 3 (6)
                # after culling. We specifically avoid culling, as this
                # would severly mess up the shapes involved.
                if q.shape[1] == 2:
                    norb_new = max(t2int(norb_new), t2int(nat_new) * defaults.DP_SHAPE)
                elif q.shape[1] == 3:
                    norb_new = max(t2int(norb_new), t2int(nat_new) * defaults.QP_SHAPE)

                # If the largest system was culled from batch, cut the
                # properties down to the new size to remove superfluous
                # padding values
                if norb > norb_new:
                    slicers["orbital"] = [slice(0, i) for i in [norb_new]]
                    norb = norb_new
                    _norb = _norb_new
                    if self.config.scp_mode == labels.SCP_MODE_FOCK:
                        mpdim = norb
                if nsh > nsh_new:
                    slicers["shell"] = [slice(0, i) for i in [nsh_new]]
                    nsh = nsh_new
                if nat > nat_new:
                    slicers["atom"] = [slice(0, i) for i in [nat_new]]
                    nat = nat_new

                # cull SCF variables
                self._data.cull(conv, slicers=slicers)

                # cull local variables
                q = q[~conv, :mpdim, :norb]
                idxs = idxs[~conv]

                if self._data.charges["mono"] is not None:
                    self._data.charges["mono"] = torch.Size((len(idxs), int(_norb)))
                if self._data.charges["dipole"] is not None:
                    self._data.charges["dipole"] = torch.Size(
                        (len(idxs), int(nat), defaults.DP_SHAPE)
                    )
                if self._data.charges["quad"] is not None:
                    self._data.charges["quad"] = torch.Size(
                        (len(idxs), int(nat), defaults.QP_SHAPE)
                    )
                if self._data.potential["mono"] is not None:
                    self._data.potential["mono"] = torch.Size((len(idxs), int(_norb)))
                if self._data.potential["dipole"] is not None:
                    self._data.potential["dipole"] = torch.Size(
                        (len(idxs), int(nat), defaults.DP_SHAPE)
                    )
                if self._data.potential["quad"] is not None:
                    self._data.potential["quad"] = torch.Size(
                        (len(idxs), int(nat), defaults.QP_SHAPE)
                    )

                # cull mixer (only contains orbital resolved properties)
                self.mixer.cull(conv, slicers=slicers["orbital"], mpdim=int(mpdim))

        # handle unconverged case (`maxiter` iterations)
        else:
            msg = (
                f"\nSCF does not converge after '{maxiter}' cycles using "
                f"'{self.mixer.label}' mixing with a damping factor of "
                f"'{self.mixer.options['damp']}'."
            )
            if self.config.force_convergence is True:
                raise exceptions.SCFConvergenceError(msg)

            # collect unconverged indices with convergence tracker; charges
            # are already culled, and hence, require no further indexing
            idxs = torch.arange(guess.size(0))
            iconv = idxs[~converged]
            q_converged[iconv, ..., :norb] = q

            # if nothing converged, skip culling
            if (~converged).all():
                culled = False

            # at least issue a helpful warning
            msg_converged = (
                "\nForced convergence is turned off. The calculation will "
                "continue with the current unconverged charges."
                f"\nIn total, {len(iconv)} systems did not converge "
                f"({iconv.tolist()}), and {len(idxs[converged])} converged "
                f"({idxs[converged].tolist()})."
            )
            OutputHandler.warn(msg + msg_converged, exceptions.SCFConvergenceWarning)

        if culled:
            # write converged variables back to `self._data` for final
            # energy evaluation; if we continue with unconverged properties,
            # we first need to write the unconverged values from the
            # `_data` object back to the converged variable before saving it
            # for the final energy evaluation
            if not converged.all():
                idxs = torch.arange(guess.size(0))
                iconv = idxs[~converged]

                cevals[iconv, :norb] = self._data.evals
                cevecs[iconv, :norb, :norb] = self._data.evecs
                ce[iconv, :norb] = self._data.energy
                ch[iconv, :norb, :norb] = self._data.hamiltonian
                co[iconv, :norb, :norb] = self._data.occupation
                cd[iconv, :norb, :norb] = self._data.density

            self._data.evals = cevals
            self._data.evecs = cevecs
            self._data.energy = ce
            self._data.hamiltonian = ch
            self._data.occupation = co
            self._data.density = cd
            self._data.charges = charges_data
            self._data.potential = potential_data

            # write culled variables (that did not change throughout the
            # SCF) back to `self._data` for the final energy evaluation
            self._data.n0 = n0
            self._data.numbers = numbers

            self._data.ints.run_checks = False
            self._data.ints.overlap = overlap
            self._data.ints.hcore = hcore
            if self._data.ints.dipole is not None and dipole is not None:
                self._data.ints.dipole = dipole
            if self._data.ints.quadrupole is not None and quad is not None:
                self._data.ints.quadrupole = quad
            self._data.ints.run_checks = True

            # reset IndexHelper and caches which were culled as well
            self._data.ihelp.restore()
            self._data.cache.restore()

        if return_charges is True:
            return self.converged_to_charges(q_converged)
        return q_converged
