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

from .._types import Any, SCFResult, Slicers, Tensor
from ..basis import IndexHelper
from ..constants import defaults
from ..exlibs.xitorch import optimize as xto
from ..interaction import InteractionList
from ..utils import SCFConvergenceError, SCFConvergenceWarning
from .base import BaseTSCF, BaseXSCF
from .guess import get_guess
from .mixer import Anderson, Mixer, Simple

from .conversions import (
    converged_to_charges,
    charges_to_potential,
    potential_to_hamiltonian,
)
from .iterations import iter_options
from .data import _Data
from .energies import get_energy, get_electronic_free_energy
from .config import SCF_Config


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

    def scf(self, guess: Tensor) -> Tensor:
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


def scf_pure(
    guess: Tensor,
    data: _Data,
    interactions: InteractionList,
    cfg: SCF_Config,
    fcn: callable,
) -> Tensor:
    """
    Self-consistent field method, which can be used to obtain a
    self-consistent solution for a given Hamiltonian.

    The method makes use of the implicit function theorem. Hence, the
    derivatives of the iterative procedure are only calculated from the
    equilibrium solution, i.e., the gradient must not be tracked through all
    iterations.

    The implementation is based on `xitorch <https://xitorch.readthedocs.io>`__,
    which appears to be abandoned and unmaintained at the time of
    writing, but still provides a reasonably good implementation of the
    iterative solver required for the self-consistent field iterations.

    This method is implemented as a pure function in order to avoid memory remnants
    of pytorch autograd graph to cause RAM issues.
    """

    # The initial guess is an "arbitrary" tensor, and hence not part of AD computational graph.
    # NOTE: This leads to not entering xitorch._RootFinder.backward() at all during a 
    #       loss.backward() call. However, then the position tensor does receive gradient.
    guess = guess.detach() 

    q_converged = xto.equilibrium(
        fcn=fcn,
        y0=guess,
        params=[data, cfg, interactions],
        bck_options={**cfg.bck_options},
        **cfg.fwd_options,
    )
    # NOTE: Entering 
    #         A) a guess that requires grad or 
    #         B) a data.tensor into params or 
    #         C) a tensor into params that requires grad 
    #       leads to memory remnants and thus a RAM leak.

    # To reconnect the H0 energy with the computational graph, we
    # compute one extra SCF cycle with strong damping.
    # Note that this is not required for SCF with full gradient tracking.
    # (see https://github.com/grimme-lab/xtbML/issues/124)
    if cfg.scp_mode in ("charge", "charges"):
        mixer = Simple({**cfg.fwd_options, "damp": 1e-4})
        q_new = fcn(q_converged, data, cfg, interactions)
        q_converged = mixer.iter(q_new, q_converged)

    return converged_to_charges(q_converged, data, cfg)


def run_scf(
    data: _Data,
    interactions: InteractionList,
    cfg: SCF_Config,
    charges: Tensor | None = None,
) -> SCFResult:
    """
    Run the self-consistent iterations until a stationary solution is
    reached.

    Parameters
    ----------
    data: _Data
        Storage for tensors which become part of autograd graph within SCF cycles.
    interactions : InteractionList
        Collection of `Interation` objects.
    cfg: SCF_Config
        Dataclass containing configuration for SCF iterations.
    charges : Tensor, optional
        Initial orbital charges vector. If `None` is given (default), a
        zero vector is used.

    Returns
    -------
    Tensor
        Converged orbital charges vector.
    """

    if charges is None:
        charges = torch.zeros_like(data.occupation)

    if cfg.scp_mode in ("charge", "charges"):
        guess = charges
    elif cfg.scp_mode == "potential":
        guess = charges_to_potential(charges, interactions, data)
    elif cfg.scp_mode == "fock":
        potential = charges_to_potential(charges, interactions, data)
        guess = potential_to_hamiltonian(potential, data)
    else:
        raise ValueError(f"Unknown convergence target (SCP mode) '{cfg.scp_mode}'.")

    # choose physical value to equilibrate (e.g. iterate_potential)
    fcn = iter_options[cfg.scp_mode]

    if cfg.scf_options["verbosity"] > 0:
        print(
            f"\n{'iter':<5} {'energy':<24} {'energy change':<15}"
            f"{'P norm change':<15} {'charge change':<15}"
        )
        print(77 * "-")

    # main SCF function (mixing)
    charges = scf_pure(guess, data, interactions, cfg, fcn)

    if cfg.scf_options["verbosity"] > 0:
        print(77 * "-")

    # evaluate final energy
    energy = get_energy(charges, data, interactions)
    fenergy = get_electronic_free_energy(data, cfg)

    # break circular graph references to free `_Data` object and hence memory
    density, hamiltonian, _, evals, evecs = data.clean()

    return {
        "charges": charges,
        "coefficients": evecs,
        "density": density,
        "emo": evals,
        "energy": energy,
        "fenergy": fenergy,
        "hamiltonian": hamiltonian,
        "occupation": data.occupation,
        "potential": charges_to_potential(charges, interactions, data),
    }


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

    def scf(self, guess: Tensor) -> Tensor:
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

        # batched SCF with culling
        else:
            culled = True

            # Initialize variables that change throughout the SCF. Later, we
            # fill these with the converged values and simultaneously cull
            # them from `self._data`
            q_converged = torch.zeros_like(guess)
            ce = torch.zeros_like(guess)
            ch = torch.zeros_like(self._data.hamiltonian)
            cevals = torch.zeros_like(self._data.evals)
            cevecs = torch.zeros_like(self._data.evecs)
            co = torch.zeros_like(self._data.occupation)
            overlap = self._data.overlap
            n0 = self._data.n0
            hcore = self._data.hcore
            numbers = self._data.numbers

            # indices for systems in batch, required for culling
            idxs = torch.arange(guess.size(0))

            # tracker for converged systems
            converged = torch.full(idxs.shape, False)

            # maximum number of orbitals in batch
            norb = self._data.ihelp.nao
            nsh = self._data.ihelp.nsh
            nat = self._data.ihelp.nat

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
                    q_converged[iconv, :norb] = q[conv, :]
                    ch[iconv, :norb, :norb] = self._data.hamiltonian[conv, :, :]
                    cevecs[iconv, :norb, :norb] = self._data.evecs[conv, :, :]
                    cevals[iconv, :norb] = self._data.evals[conv, :]
                    ce[iconv, :norb] = self._data.energy[conv, :]
                    co[iconv, :norb, :norb] = self._data.occupation[conv, :, :]

                    # update convergence tracker
                    converged[iconv] = True

                    # end SCF if all systems are converged
                    if conv.all():
                        break

                    # cull `orbitals_per_shell` (`shells_per_atom`) to
                    # calculate maximum number of orbitals (shells), which
                    # corresponds to the maximum padding
                    norb_new = (
                        self._data.ihelp.orbitals_per_shell[~conv, ...].sum(-1).max()
                    )
                    nsh_new = self._data.ihelp.shells_per_atom[~conv, ...].sum(-1).max()
                    nat_new = self._data.numbers[~conv, ...].count_nonzero(dim=-1).max()

                    # if the largest system was culled from batch, cut the
                    # properties down to the new size to remove superfluous
                    # padding values
                    slicers: Slicers = {
                        "orbital": (...,),
                        "shell": (...,),
                        "atom": (...,),
                    }
                    if norb != norb_new:
                        slicers["orbital"] = [slice(0, i) for i in [norb_new]]
                        norb = norb_new
                    if nsh != nsh_new:
                        slicers["shell"] = [slice(0, i) for i in [nsh_new]]
                        nsh = nsh_new
                    if nat != nat_new:
                        slicers["atom"] = [slice(0, i) for i in [nat_new]]
                        nat = nat_new

                    # cull SCF variables
                    self._data.cull(conv, slicers=slicers)

                    # cull local variables
                    q = q[~conv, :norb]
                    idxs = idxs[~conv]

                    # cull mixer (only contains orbital resolved properties)
                    mixer.cull(conv, slicers=slicers["orbital"])

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
                q_converged[iconv, :] = q

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

                    cevals[iconv, :] = self._data.evals
                    cevecs[iconv, :, :] = self._data.evecs
                    ce[iconv, :] = self._data.energy
                    ch[iconv, :, :] = self._data.hamiltonian
                    co[iconv, :, :] = self._data.occupation

                self._data.evals = cevals
                self._data.evecs = cevecs
                self._data.energy = ce
                self._data.hamiltonian = ch
                self._data.occupation = co

                # write culled variables (that did not change throughout the
                # SCF) back to `self._data` for the final energy evaluation
                self._data.n0 = n0
                self._data.hcore = hcore
                self._data.overlap = overlap
                self._data.numbers = numbers

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

    def __call__(self, charges: Tensor | None = None) -> dict[str, Tensor]:
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
            charges = torch.zeros_like(self._data.occupation)

        # calculate charges in SCF without gradient tracking
        with torch.no_grad():
            q_conv = self.scf(charges)

        # SCF step with gradient using converged result as "perfect" guess
        out = (
            self.iterate_potential(self.charges_to_potential(q_conv))
            if self.use_potential
            else self.iterate_charges(q_conv)
        )
        charges = self.potential_to_charges(out) if self.use_potential else out

        # Check consistency between SCF solution and single step.
        # Especially for elements and their ions, the SCF may oscillate and the
        # single step for the gradient may differ from the converged solution.
        if (
            torch.linalg.vector_norm(q_conv - charges)
            > sqrt(torch.finfo(self.dtype).eps) * 10
        ).any():
            warnings.warn(
                "The single SCF step differs from the converged solution. "
                "Re-calculating with full gradient tracking!"
            )
            charges = self.scf(q_conv)

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
    *args: Any,
    **kwargs: Any,
) -> SCFResult:
    """converged_to_charges
    Obtain self-consistent solution for a given Hamiltonian.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of the system.
    positions : Tensor
        Positions of the system.
    chrg : Tensor
        Total charge.
    interactions : InteractionList
        Collection of `Interation` objects.
    ihelp : IndexHelper
        Index helper object.
    guess : str
        Name of the method for the initial charge guess.
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
        # A) calculate SCF equilibrium using nested objects
        # scf = SelfConsistentField

        # B) calculate SCF equilibrium using semi-pure functions

        # distinct objects containing data and configuration
        forbidden = ["bck_options", "fwd_options", "scf_options"]
        data_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden}
        data = _Data(*args, numbers=numbers, ihelp=ihelp, cache=cache, **data_kwargs)
        cfg = SCF_Config(data, **kwargs)

        charges = get_guess(numbers, positions, chrg, ihelp, guess)
        result = run_scf(data, interactions, cfg, charges)

        return result

    elif scf_mode in ("full", "full_tracking"):
        scf = SelfConsistentFieldFull
    elif scf_mode == "experimental":
        scf = SelfConsistentFieldSingleShot
    else:
        raise ValueError(f"Unknown SCF mode '{scf_mode}'.")

    charges = get_guess(numbers, positions, chrg, ihelp, guess)
    return scf(
        interactions, *args, numbers=numbers, ihelp=ihelp, cache=cache, **kwargs
    )(charges)
