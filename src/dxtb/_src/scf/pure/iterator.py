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

import torch

from dxtb import IndexHelper, OutputHandler
from dxtb._src.components.interactions import (
    Charges,
    InteractionList,
    InteractionListCache,
)
from dxtb._src.constants import defaults, labels
from dxtb._src.exlibs.xitorch import optimize as xto
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.typing import Any, Callable, Tensor
from dxtb.config import ConfigSCF

from ..mixer import Simple
from ..result import SCFResult
from .conversions import (
    charges_to_potential,
    converged_to_charges,
    potential_to_hamiltonian,
)
from .data import _Data
from .energies import get_electronic_free_energy, get_energy
from .iterations import iter_options

__all__ = ["scf_pure", "scf_wrapper", "run_scf"]


def scf_pure(
    guess: Tensor,
    data: _Data,
    interactions: InteractionList,
    cfg: ConfigSCF,
    fcn: Callable[[Tensor, _Data, ConfigSCF, InteractionList], Tensor],
) -> Charges:
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

    This method is implemented as a pure function in order to avoid memory
    remnants of the pytorch autograd graph that cause RAM issues.
    """

    # The initial guess is an "arbitrary" tensor, and hence not part of AD
    # computational graph.
    # NOTE: This leads to not entering xitorch._RootFinder.backward() at all
    # during a loss.backward() call. However, then the position tensor does
    # receive gradient.
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
    if cfg.scp_mode == labels.SCP_MODE_CHARGE:
        mixer = Simple({**cfg.fwd_options, "damp": 1e-4})
        q_new = fcn(q_converged, data, cfg, interactions)
        q_converged = mixer.iter(q_new, q_converged)

    return converged_to_charges(q_converged, data, cfg)


def scf_wrapper(
    interactions: InteractionList,
    occupation: Tensor,
    n0: Tensor,
    guessq: Tensor,
    numbers: Tensor,
    *,
    ihelp: IndexHelper,
    cache: InteractionListCache,
    integrals: IntegralMatrices,
    config: ConfigSCF,
    **kwargs: Any,
) -> SCFResult:
    # calculate SCF equilibrium using semi-pure functions

    # distinct objects containing data and configuration
    # forbidden = ["bck_options", "fwd_options", "scf_options"]
    # data_kwargs = {k: v for k, v in kwargs.items() if k not in forbidden}
    data = _Data(
        occupation=occupation,
        n0=n0,
        numbers=numbers,
        ihelp=ihelp,
        cache=cache,
        integrals=integrals,
        # **data_kwargs,
    )

    config.bck_options = {"posdef": True, **kwargs.pop("bck_options", {})}
    config.fwd_options = {
        "force_convergence": False,
        "method": "broyden1",
        "alpha": -0.5,
        "damp": config.damp,
        "f_tol": config.f_atol,
        "x_tol": config.x_atol,
        "f_rtol": float("inf"),
        "x_rtol": float("inf"),
        "maxiter": config.maxiter,
        "verbose": False,
        "line_search": False,
        **kwargs.pop("fwd_options", {}),
    }

    config.eigen_options = {"method": "exacteig", **kwargs.pop("eigen_options", {})}

    # Only infer shapes and types from _Data (no logic involved),
    # i.e. keep _Data and SCFConfig instances disjunct objects.
    config.shape = data.ints.hcore.shape

    result = run_scf(data, interactions, config, guessq)

    return result


def run_scf(
    data: _Data,
    interactions: InteractionList,
    cfg: ConfigSCF,
    charges: Tensor | Charges | None = None,
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
    cfg: SCFConfig
        Dataclass containing configuration for SCF iterations.
    charges : Tensor, optional
        Initial orbital charges vector. If ``None`` is given (default), a
        zero vector is used.

    Returns
    -------
    Tensor
        Converged orbital charges vector.
    """
    # initialize zero charges (equivalent to SAD guess)
    if charges is None:
        charges = torch.zeros_like(data.occupation)

    # initialize Charge container depending on given integrals
    if isinstance(charges, Tensor):
        charges = Charges(mono=charges, batch_mode=cfg.batch_mode)
        data.charges["mono"] = charges.mono_shape

        if data.ints.dipole is not None:
            shp = (*data.numbers.shape, defaults.DP_SHAPE)
            zeros = torch.zeros(
                shp, device=charges.mono.device, dtype=charges.mono.dtype
            )
            charges.dipole = zeros
            data.charges["dipole"] = charges.dipole_shape

        if data.ints.quadrupole is not None:
            shp = (*data.numbers.shape, defaults.QP_SHAPE)
            zeros = torch.zeros(
                shp, device=charges.mono.device, dtype=charges.mono.dtype
            )
            charges.quad = zeros
            data.charges["quad"] = charges.quad_shape

    if cfg.scp_mode == labels.SCP_MODE_CHARGE:
        guess = charges.as_tensor()
    elif cfg.scp_mode == labels.SCP_MODE_POTENTIAL:
        potential = charges_to_potential(charges, interactions, data)
        guess = potential.as_tensor()
    elif cfg.scp_mode == labels.SCP_MODE_FOCK:
        potential = charges_to_potential(charges, interactions, data)
        guess = potential_to_hamiltonian(potential, data)
    else:
        raise ValueError(f"Unknown convergence target (SCP mode) '{cfg.scp_mode}'.")

    OutputHandler.write_stdout(
        f"\n{'iter':<5} {'Energy':<24} {'Delta E':<16}"
        f"{'Delta Pnorm':<15} {'Delta q':<15}",
        v=3,
    )
    OutputHandler.write_stdout(77 * "-", v=3)

    # choose physical value to equilibrate (e.g. iterate_potential)
    fcn = iter_options[cfg.scp_mode]

    # main SCF function (mixing)
    charges = scf_pure(guess, data, interactions, cfg, fcn)

    OutputHandler.write_stdout(77 * "-", v=3)
    OutputHandler.write_stdout("", v=3)

    # evaluate final energy
    charges.nullify_padding()
    energy = get_energy(charges, data, interactions)
    fenergy = get_electronic_free_energy(data, cfg)

    # break circular graph references to free `_Data` object and hence memory
    density, hamiltonian, _, evals, evecs, occupation = data.clean()

    return {
        "charges": charges,
        "coefficients": evecs,
        "density": density,
        "emo": evals,
        "energy": energy,
        "fenergy": fenergy,
        "hamiltonian": hamiltonian,
        "occupation": occupation,
        "potential": charges_to_potential(charges, interactions, data),
        "iterations": data.iter,
    }
