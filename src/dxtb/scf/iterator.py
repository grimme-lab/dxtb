"""
Self-consistent field iteration
===============================

Provides implementation of self consistent field iterations for the xTB
Hamiltonian. The iterations are not like in ab initio calculations expressed in
the density matrix and the derivative of the energy w.r.t. to the density
matrix, i.e. the Hamiltonian, but the Mulliken populations (or partial charges)
of the respective orbitals as well as the derivative of the energy w.r.t. to
those populations, i.e. the potential vector.
"""
from __future__ import annotations

import warnings
from math import sqrt

import torch

from .._types import Any, Slicers, Tensor
from ..basis import IndexHelper
from ..constants import defaults
from ..exlibs.xitorch import optimize as xto
from ..integral import IntegralMatrices
from ..interaction import Charges, InteractionList
from ..utils import SCFConvergenceError, SCFConvergenceWarning, t2int
from .base import BaseTSCF, BaseXSCF, SCFResult
from .guess import get_guess
from .mixer import Anderson, Mixer, Simple


class SelfConsistentField(BaseXSCF):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    The default class makes use of the implicit function theorem. Hence, the
    derivatives of the iterative procedure are only calculated from the
    equilibrium solution, i.e., the gradient must not be tracked through all
    iterations.

    The implementation is based on `xitorch <https://xitorch.readthedocs.io>`__,
    which appears to be abandoned and unmaintained at the time of
    writing, but still provides a reasonably good implementation of the
    iterative solver required for the self-consistent field iterations.
    """

    def scf(self, guess: Tensor) -> Charges:
        fcn = self._fcn

        q_converged = xto.equilibrium(
            fcn=fcn,
            y0=guess,
            bck_options={**self.bck_options},
            **self.fwd_options,
        )

        # To reconnect the H0 energy with the computational graph, we
        # compute one extra SCF cycle with strong damping.
        # Note that this is not required for SCF with full gradient tracking.
        # (see https://github.com/grimme-lab/xtbML/issues/124)
        if self.scp_mode in ("charge", "charges"):
            mixer = Simple({**self.fwd_options, "damp": 1e-4})
            q_new = fcn(q_converged)
            q_converged = mixer.iter(q_new, q_converged)

        return self.converged_to_charges(q_converged)


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

    def scf(self, guess: Tensor) -> Charges:
        # "SCF function"
        fcn = self._fcn

        mixer = self.scf_options["mixer"]
        maxiter = self.fwd_options["maxiter"]

        # initialize the correct mixer with tolerances etc.
        if isinstance(mixer, str):
            mixers = {"anderson": Anderson, "simple": Simple}
            if mixer.casefold() not in mixers:
                raise ValueError(f"Unknown mixer '{mixer}'.")
            mixer: Mixer = mixers[mixer.casefold()](
                self.fwd_options, is_batch=self.batched
            )

        q = guess

        # single-system (non-batched) case, which does not require culling
        if not self.batched:
            for _ in range(maxiter):
                q_new = fcn(q)
                q = mixer.iter(q_new, q)

                if mixer.converged:
                    q_converged = q
                    break

            else:
                msg = (
                    f"\nSCF does not converge after {maxiter} cycles using "
                    f"{mixer.label} mixing with a damping factor of "
                    f"{mixer.options['damp']}."
                )
                if self.fwd_options["force_convergence"]:
                    raise SCFConvergenceError(msg)

                # only issue warning, return anyway
                warnings.warn(msg, SCFConvergenceWarning)
                q_converged = q

            return self.converged_to_charges(q_converged)

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
            q = mixer.iter(q_new, q)

            conv = mixer.converged
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
                    if self.scp_mode in ("fock", "fockian"):
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
                mixer.cull(conv, slicers=slicers["orbital"], mpdim=int(mpdim))

        # handle unconverged case (`maxiter` iterations)
        else:
            msg = (
                f"\nSCF does not converge after '{maxiter}' cycles using "
                f"'{mixer.label}' mixing with a damping factor of "
                f"'{mixer.options['damp']}'."
            )
            if self.fwd_options["force_convergence"]:
                raise SCFConvergenceError(msg)

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
            warnings.warn(msg + msg_converged, SCFConvergenceWarning)

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

        return self.converged_to_charges(q_converged)


class SelfConsistentFieldSingleShot(SelfConsistentFieldFull):
    """
    Self-consistent field iterator, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    .. warning:

        Do not use in production. The gradients of the single-shot method
        are not exact (derivative w.r.t. the input features is missing).

    The single-shot gradient tracking was a first idea to reduce time and memory
    consumption in the SCF. Here, the SCF is performed outside of the purview
    of the computational graph. Only after convergence, an additional SCF step
    with enabled gradient tracking is performed to reconnect to the autograd
    engine. However, this approach is not correct as a derivative w.r.t. the
    input features is missing. Apparently, the deviations are small if the
    input features do not change much.
    """

    def __call__(self, charges: Charges | Tensor | None = None) -> SCFResult:
        """
        Run the self-consistent iterations until a stationary solution is reached

        Parameters
        ----------
        charges : Tensor, optional
            Initial orbital charges vector.

        Returns
        -------
        Tensor
            Converged orbital charges vector.
        """

        if charges is None:
            charges = Charges(mono=torch.zeros_like(self._data.occupation))
        if isinstance(charges, Tensor):
            charges = Charges(mono=charges)

        if self.scp_mode in ("charge", "charges"):
            guess = charges.as_tensor()
        elif self.scp_mode == "potential":
            potential = self.charges_to_potential(charges)
            guess = potential.as_tensor()
        elif self.scp_mode == "fock":
            potential = self.charges_to_potential(charges)
            guess = self.potential_to_hamiltonian(potential)
        else:
            raise ValueError(
                f"Unknown convergence target (SCP mode) '{self.scp_mode}'."
            )

        # calculate charges in SCF without gradient tracking
        with torch.no_grad():
            scp_conv = self.scf(guess).as_tensor()

        # initialize the correct mixer with tolerances etc.
        mixer = self.scf_options["mixer"]
        if isinstance(mixer, str):
            mixers = {"anderson": Anderson, "simple": Simple}
            if mixer.casefold() not in mixers:
                raise ValueError(f"Unknown mixer '{mixer}'.")

            # select and init mixer
            mixer: Mixer = mixers[mixer.casefold()](
                self.fwd_options, is_batch=self.batched
            )

        # SCF step with gradient using converged result as "perfect" guess
        scp_new = self._fcn(scp_conv)
        scp = mixer.iter(scp_new, scp_conv)
        scp = self.converged_to_charges(scp)

        # Check consistency between SCF solution and single step.
        # Especially for elements and their ions, the SCF may oscillate and the
        # single step for the gradient may differ from the converged solution.
        if (
            torch.linalg.vector_norm(scp_conv - scp)
            > sqrt(torch.finfo(self.dtype).eps) * 10
        ).any():
            warnings.warn(
                "The single SCF step differs from the converged solution. "
                "Re-calculating with full gradient tracking!"
            )
            charges = self.scf(scp_conv)

        charges.nullify_padding()
        energy = self.get_energy(charges)
        fenergy = self.get_electronic_free_energy()

        return {
            "charges": charges,
            "coefficients": self._data.evecs,
            "density": self._data.density,
            "emo": self._data.evals,
            "energy": energy,
            "fenergy": fenergy,
            "hamiltonian": self._data.hamiltonian,
            "occupation": self._data.occupation,
            "potential": self.charges_to_potential(charges),
        }


def solve(
    numbers: Tensor,
    positions: Tensor,
    chrg: Tensor,
    interactions: InteractionList,
    cache: InteractionList.Cache,
    ihelp: IndexHelper,
    guess: str,
    integrals: IntegralMatrices,
    *args: Any,
    **kwargs: Any,
) -> SCFResult:
    """
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system (nat, 3).
    chrg : Tensor
        Total charge.
    interactions : InteractionList
        Collection of `Interation` objects.
    ihelp : IndexHelper
        Index helper object.
    guess : str
        Name of the method for the initial charge guess.
    integrals : Integrals
        Container for all integrals.
    args : Tuple
        Positional arguments to pass to the engine.
    kwargs : dict
        Keyword arguments to pass to the engine.

    Returns
    -------
    Tensor
        Orbital-resolved partial charges vector.
    """
    scf_mode = kwargs["scf_options"].get("scf_mode", defaults.SCF_MODE)
    if scf_mode in ("default", "implicit"):
        scf = SelfConsistentField
    elif scf_mode in ("full", "full_tracking"):
        scf = SelfConsistentFieldFull
    elif scf_mode == "experimental":
        scf = SelfConsistentFieldSingleShot
    else:
        raise ValueError(f"Unknown SCF mode '{scf_mode}'.")

    charges = get_guess(numbers, positions, chrg, ihelp, guess)
    return scf(
        interactions,
        *args,
        numbers=numbers,
        ihelp=ihelp,
        cache=cache,
        integrals=integrals,
        **kwargs,
    )(charges)
