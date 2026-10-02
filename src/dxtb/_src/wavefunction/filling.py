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
Wavefunction: Filling
=====================

Handle the occupation of the orbitals with electrons.

Parts of the Fermi smearing are taken from https://github.com/tbmalt/tbmalt.
The Fermi energy search follows the Fermi filling of tblite after pull request
#385 (https://github.com/tblite/tblite/pull/385, commit ``e437cde``), with
additions for batches, fractional electrons and derivatives (see
:func:`get_fermi_occupation`). The derivatives of the occupations are obtained
by differentiating Newton steps after a detached solve, an instance of
one-step differentiation [Bolte2023]_ that is extended here to higher orders.

References
----------
.. [Bolte2023] J. Bolte, E. Pauwels, S. Vaiter. One-step differentiation of
   iterative algorithms. 37th Conference on Neural Information Processing
   Systems (NeurIPS 2023). Corollary 2 (vanishing Jacobian at the fixed point)
   and Corollary 3 (Newton's method, quadratic convergence) prove the
   first-order result. The paper contains no statement about higher orders.
"""

from __future__ import annotations

import warnings
from numbers import Integral

import torch
from tad_mctc.convert import any_to_tensor

from dxtb._src.constants import defaults
from dxtb._src.typing import Tensor

__all__ = [
    "get_alpha_beta_occupation",
    "get_aufbau_occupation",
    "get_fermi_energy",
    "get_fermi_occupation",
]


# Number of iterations of the Fermi energy search between host
# synchronizations (convergence checks).
_CHECK_EVERY = 8

# Highest order of the derivatives that is still accurate in single precision.
# Higher orders of `diff_order` (`get_fermi_occupation`) warn outside of
# double precision.
_MAX_SINGLE_DIFF_ORDER = 3

# Newton steps of the converged Fermi energy on the log-space residual (see
# `_polish`). In a gap, the first one corrects the position of the Fermi
# energy between the orbitals (the residual is almost linear there). Two steps
# leave relative errors of 1e-9 in the derivatives with respect to the
# electrons in the widest gaps (measured), three reach machine precision.
_POLISH_STEPS = 3

# A differentiable Newton step is only taken if it is at most this many kT
# long. At the polished start, the step is the (tiny) deviation of the start
# from the root. It is a safety net against a start that is not converged,
# from which Newton may overshoot by orders of magnitude in the tail of the
# Fermi function and corrupt the occupations. The Fermi energy is kept in that
# case (as for a vanishing derivative).
_MAX_DIFF_STEP_KT = 3.0


def _integer_tol(dtype: torch.dtype) -> float:
    """
    Tolerance to decide if the number of electrons (or a cumulative
    occupation) is an integer, i.e., if the orbital is fully occupied.
    """
    return torch.finfo(dtype).resolution * 5


def _read_flags(x: Tensor) -> list[bool]:
    """
    Read a 1D boolean tensor on the host with a single synchronization.

    ``Tensor.tolist`` needs a data pointer, which tensors within
    `torch.func.jacrev` do not have in some PyTorch versions (e.g., 2.4), not
    even after peeling off the wrappers. ``Tensor.item`` goes through the
    dispatcher and works, hence the flags are packed into one integer. The
    search is not differentiated, i.e., the values are all that is needed.

    Parameters
    ----------
    x : Tensor
        Boolean tensor with at most 63 elements.

    Returns
    -------
    list[bool]
        Values of the tensor.
    """
    n = x.numel()
    assert n <= 63, "Flags do not fit into a 64-bit integer."

    weights = 2 ** torch.arange(n, device=x.device, dtype=torch.int64)
    packed = int((x.to(torch.int64) * weights).sum().item())
    return [bool((packed >> i) & 1) for i in range(n)]


def _sqrttiny(dtype: torch.dtype) -> float:
    """
    Smallest derivative of the Fermi function that is still divided by.
    """
    return torch.finfo(dtype).tiny ** 0.5


def _diff_floor(dtype: torch.dtype, steps: int) -> float:
    """
    Smallest derivative of the number of electrons ``g' = sum f (1 - f) / kT``
    that the differentiable Newton steps divide by.

    The ``m``-th derivative with respect to the number of electrons grows
    like ``g'**(1 - m)`` in a gap, i.e., it overflows for ``g' < tiny**(1/m)``
    (measured, also in the backward pass). ``k`` steps are exact through
    order ``2**k - 1``. With this floor, the derivatives through order
    ``2**k`` stay finite, so the first inexact order is also finite. The
    factor ``1 / eps`` is a margin for the prefactors.
    """
    info = torch.finfo(dtype)
    return (info.tiny / info.eps) ** (1.0 / 2**steps)


def _diff_steps(diff_order: int) -> int:
    """
    Number of differentiable Newton steps for derivatives up to `diff_order`.

    ``k`` steps are exact through order ``2**k - 1``, i.e., the smallest ``k``
    with ``2**k - 1 >= diff_order`` is ``ceil(log2(diff_order + 1))``, which is
    the bit length of `diff_order`.

    Parameters
    ----------
    diff_order : int
        Highest order of the derivatives that must be exact.

    Returns
    -------
    int
        Number of Newton steps (0 for order 0, 1 for order 1, 2 for orders 2
        and 3, 3 for orders 4 to 7, ...).

    Raises
    ------
    TypeError
        `diff_order` is not an integer.
    ValueError
        `diff_order` is negative.
    """
    if isinstance(diff_order, bool) or not isinstance(diff_order, Integral):
        raise TypeError(
            f"The derivative order must be an integer (got {diff_order!r})."
        )
    if diff_order < 0:
        raise ValueError(
            f"The derivative order must not be negative ({diff_order})."
        )
    return int(diff_order).bit_length()


def get_alpha_beta_occupation(
    nel: Tensor, uhf: Tensor | float | int | list[int] | None = None
) -> Tensor:
    """
    Generate alpha and beta electrons from total number of electrons.

    Parameters
    ----------
    nel : Tensor
        Total number of electrons.
    uhf : Tensor | int | list[int] | None
        Number of unpaired electrons. If ``None``, spin is figured out
        automatically.

    Returns
    -------
    Tensor
        Alpha (first column, 0 index) and beta (second column, 1 index)
        electrons.

    Raises
    ------
    ValueError
        Number of electrons and unpaired electrons does not match.

    Note
    ----
    The number of electrons is rounded to integers via `torch.round` for
    numerical stability, i.e., non-integer electrons are not supported.
    """
    if uhf is not None:
        if isinstance(uhf, (list, int, float)):
            uhf = torch.tensor(uhf, device=nel.device, dtype=nel.dtype)
        else:
            uhf = uhf.type(nel.dtype).to(nel.device)

        if uhf.shape != nel.shape:
            raise RuntimeError(
                f"Shape mismatch for unpaired electrons ({uhf.shape}) and "
                f"number of electrons ({nel.shape})."
            )

        if (uhf > nel.round()).any():
            raise ValueError(
                f"Number of unpaired electrons ({uhf}) larger than "
                f"number of electrons ({nel})."
            )

        # odd/even spin and even/odd number of electrons
        if (torch.remainder(uhf, 2) != torch.remainder(nel.round(), 2)).any():
            raise ValueError(
                f"Odd (even) number of unpaired electrons ({uhf}) but even "
                f"(odd) number of electrons ({nel}) given."
            )
    else:
        # set to zero and figure out via remainder
        uhf = torch.zeros_like(nel)

    nel = torch.atleast_1d(nel)
    uhf = torch.atleast_1d(uhf)
    assert isinstance(uhf, Tensor)

    nuhf = torch.where(
        torch.remainder(uhf, 2) == torch.remainder(nel.round(), 2),
        uhf,
        torch.remainder(nel.round(), 2),
    )

    diff = torch.minimum(nuhf, nel)
    nb = (nel - diff) / 2.0
    na = nb + diff

    return torch.cat([na, nb], dim=-1)


def get_aufbau_occupation(norb: Tensor, nel: Tensor) -> Tensor:
    """
    Set occupation numbers according to the aufbau principle.
    The number of electrons is a real number and can be fractional.
    Orbitals beyond the number of available orbitals `norb` of a system
    (padding in a batch) are never occupied.

    Parameters
    ----------
    norb : Tensor
        Number of available orbitals.
    nel : Tensor
        Number of electrons.

    Returns
    -------
    Tensor
        Occupation numbers.

    Examples
    --------
    >>> get_aufbau_occupation(torch.tensor(5), torch.tensor(1.))
    tensor([1., 0., 0., 0., 0.])
    >>> get_aufbau_occupation(torch.tensor([8, 8, 5]), torch.tensor([2., 3., 1.]))
    tensor([[1., 1., 0., 0., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0.]])
    >>> nel, norb = torch.tensor([2.0, 3.5, 1.5]), torch.tensor([4, 4, 2])
    >>> occ = get_aufbau_occupation(norb, nel)
    >>> occ
    tensor([[1.0000, 1.0000, 0.0000, 0.0000],
            [1.0000, 1.0000, 1.0000, 0.5000],
            [1.0000, 0.5000, 0.0000, 0.0000]])
    >>> all(nel == occ.sum(-1))
    True

    .. code-block:: python

        import torch
        from dxtb.wavefunction import get_aufbau_occupation

        # 1 electron in 5 orbitals
        r1 = get_aufbau_occupation(torch.tensor(5), torch.tensor(1.))

        print(r1)
        # Output: tensor([1., 0., 0., 0., 0.])


        # Multiple orbitals and different electron counts
        r2 = get_aufbau_occupation(
            torch.tensor([8, 8, 5]), torch.tensor([2., 3., 1.])
        )

        print(r2)
        # Output: tensor([[1., 1., 0., 0., 0., 0., 0., 0.],
        #                 [1., 1., 1., 0., 0., 0., 0., 0.],
        #                 [1., 0., 0., 0., 0., 0., 0., 0.]])

        # Fractional electron numbers in multiple orbitals
        nel, norb = torch.tensor([2.0, 3.5, 1.5]), torch.tensor([4, 4, 2])
        occ = get_aufbau_occupation(norb, nel)

        print(occ)
        # Output: tensor([[1.0000, 1.0000, 0.0000, 0.0000],
        #                 [1.0000, 1.0000, 1.0000, 0.5000],
        #                 [1.0000, 0.5000, 0.0000, 0.0000]])

        # Check if the total number of electrons matches the sum of occupation
        print(all(nel == occ.sum(-1)))  # True
    """

    # electrons fill the available orbitals, orbitals of smaller systems in a
    # batch (`norb` per system) are padding
    nmax = int(torch.max(norb).item())
    valid = None  # [b, (1,) nmax]: True for orbitals that exist
    if norb.dim() > 0:
        # `norb` has one entry per system ([b]), `nel` may have additional
        # (channel) dimensions between the systems and the orbitals
        # ([b, 2]), i.e., `norb` becomes [b, 1, 1] and the orbital indices
        # [nmax] are broadcast against it
        extra = max(nel.dim() - norb.dim(), 0)
        shape = (*norb.shape, *([1] * (extra + 1)))
        valid = torch.arange(nmax, device=nel.device) < norb.reshape(shape)

    # the trailing dimension of the electrons ([b, 2, 1]) is broadcast against
    # the orbitals ([b, 2, nmax]); a scalar `nel` with a batch of `norb` fills
    # every system with the same number of electrons ([b, nmax])
    return _aufbau_occupation(
        nel.unsqueeze(-1), nmax, valid, nel.dtype, nel.device
    )


def get_fermi_energy(
    nel: Tensor, emo: Tensor, mask: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """
    Get Fermi energy as midpoint between the HOMO and LUMO.

    The orbital energies `emo` and the `mask` must already have the correct
    shape for using alpha/beta electron channels. Spreading to the channels can
    be done with `x.unsqueeze(-2).expand([*nel.shape, -1])`.

    Parameters
    ----------
    nel : Tensor
        Number of electrons per channel (shape ``[b, 2]``, the batch dimension
        ``b`` is optional).
    emo : Tensor
        Orbital energies (shape ``[b, 2, n]``, the same for both channels).
    mask : Tensor | None, optional
        Mask from orbitals to avoid reading padding as LUMO for elements
        without LUMO due to minimal basis (shape ``[b, 2, n]``).

    Returns
    -------
    tuple[Tensor, Tensor]
        Fermi energy (shape ``[b, 2]``) and index of HOMO (shape
        ``[b, 2, 1]``).
    """
    zero = torch.tensor(0.0, device=emo.device, dtype=emo.dtype)

    # cumulative number of orbitals minus the number of electrons of the
    # channel ([b, 2, n]); `nel` gets a trailing dimension for the orbitals
    occ = torch.ones_like(emo)
    occ_cs = occ.cumsum(-1) - nel.unsqueeze(-1)

    # transition: negative values indicate end of occupied orbitals
    temp = occ_cs >= -_integer_tol(emo.dtype)

    # index of first non-negative value and unsqueeze for stacking;
    # stacking will happen along that dim
    # (shape [b, 2, 1])
    homo = torch.argmax(temp.type(torch.long), dim=-1).unsqueeze(-1)

    # some atoms (e.g., He) do not have a LUMO because of the valence basis and
    # the LUMO index becomes larger than No. MOs
    lumo_missing = occ.sum(-1, keepdim=True) - 1 <= homo
    # indices of HOMO and LUMO ([b, 2, 2])
    gap = torch.where(
        lumo_missing,
        torch.cat((homo, homo), -1),  # Fermi energy becomes HOMO energy
        torch.cat((homo, homo + 1), -1),
    )

    # Fermi energy as midpoint between HOMO and LUMO ([b, 2])
    e_fermi = torch.where(
        nel != 0,  # detect empty beta channel
        torch.gather(emo, -1, gap).mean(-1),
        zero,  # no electrons yield Fermi energy of 0.0
    )

    # NOTE:
    # In batched calculations, the missing LUMO is replaced by padding, which is
    # not caught by the above `torch.where`. Consequently, the LUMO is 0.0 and
    # the Fermi energy is exactly half of the correct value. To fix this, a mask
    # from the orbitals of the IndexHelper is gathered in the same way as the
    # Fermi energy. The `prod(-1)` reduces the dimension as `mean(-1)` does.
    # Finally, multiplication by two corrects the mean, taken with E_LUMO = 0.
    if mask is not None:
        mask = torch.where(mask == 0, mask, torch.ones_like(mask))
        mask = torch.gather(mask, -1, gap).prod(-1)
        e_fermi = torch.where(mask != 0, e_fermi, e_fermi * 2.0)

    return e_fermi, homo


def get_fermi_occupation(
    nel: Tensor,
    emo: Tensor,
    kt: Tensor,
    mask: Tensor | None = None,
    thr: Tensor | float | int | None = None,
    maxiter: int = 200,
    diff_order: int | None = None,
) -> Tensor:
    """
    Set occupation numbers according to Fermi distribution.

    The Fermi energy is determined such that the occupations sum up to the
    given (possibly fractional) number of electrons `nel`. The algorithm is
    the one of tblite (Newton iteration on the number of electrons) with three
    additions that are necessary for a batched, differentiable
    implementation:

    1. The Newton iteration is safeguarded by a bisection bracket because it
       is not globally convergent, e.g., for fractional electrons. Converged
       entries of a batch are frozen. This part does not track gradients.
    2. The converged Fermi energy is polished by Newton steps on the residual
       in log space, which pins it to machine precision also in a gap, where
       any Fermi energy between the orbitals converges the number of
       electrons. This part does not track gradients either.
    3. Differentiable Newton steps from the converged Fermi energy carry its
       derivatives up to the requested order `diff_order` (see the note on
       derivatives below) and are attached to the graph afterwards.

    The orbital energies `emo` must already have the correct shape for using
    alpha/beta electron channels. Spreading to the channels can be done with
    `emo.unsqueeze(-2).expand([*nel.shape, -1])`.

    Shapes (the batch dimension ``b`` is optional, ``n`` is the number of
    orbitals including padding): the electrons `nel` are ``[b, 2]``, the
    orbital energies `emo`, the `mask` and the occupations are ``[b, 2, n]``.
    Each channel (alpha, beta) is optimized independently and the orbitals are
    at most singly occupied. Intermediate quantities per channel, such as the
    Fermi energy or the number of electrons, have a trailing singleton
    dimension (``[b, 2, 1]``) to broadcast against the orbitals.

    Parameters
    ----------
    nel : Tensor
        Number of electrons per channel (``[b, 2]``). It may be fractional,
        and the occupations are differentiable with respect to it (e.g., for
        the chemical potential or Fukui functions). A graph that must not be
        part of the occupations (e.g., the one of the previous SCF step) has
        to be detached by the caller.
    emo : Tensor
        Orbital energies (``[b, 2, n]``).
    kt : Tensor
        Electronic temperature in atomic units (scalar). It is moved to the
        device of `emo`. For ``kt == 0``, the aufbau occupation is returned.
    mask : Tensor | None, optional
        Mask for the existing orbitals (``0`` for padding) with the same
        shape as `emo`. Padded orbitals are never occupied and are not read
        as LUMO for the initial guess of the Fermi energy. Without a mask,
        padding cannot be distinguished from actual orbitals and is occupied
        if its energy is close to the Fermi energy.
    thr : Tensor | float | int | None, optional
        Threshold for the deviation of the number of electrons, by default
        ``None``, which is ``min(sqrt(eps), 1e5 * eps, 1e-4)`` of the dtype of
        `emo` (the last limit only applies in single precision). The SCF
        verifies the number of electrons to about 5e-4, i.e., larger
        thresholds may trip this check in single precision.
    maxiter : int, optional
        Maximum number of iterations for converging Fermi energy.
        Defaults to 200.
    diff_order : int | None, optional
        Highest order of the derivatives of the occupations that is exact
        (with respect to the orbital energies, the temperature, the number of
        electrons and everything upstream). ``k = ceil(log2(diff_order + 1))``
        differentiable Newton steps are attached, i.e., 0 for order 0, 1 for
        order 1, 2 for orders 2 and 3, and 3 for orders 4 to 7. The default
        ``None`` is :data:`~dxtb._src.constants.defaults.FERMI_DIFF_ORDER`
        (order 3: forces, Hessians, polarizabilities, dipole derivatives and
        first hyperpolarizabilities), set with the ``fermi_diff_order``
        option in the SCF. The value of the occupations does not depend
        on it (apart from the forward residual, see below). Each additional
        order of the derivative costs more than the additional Newton steps,
        since the nested derivatives of the graph grow exponentially. Orders
        beyond 3 need double precision.

    Returns
    -------
    Tensor
        Occupation numbers.

    Raises
    ------
    RuntimeError
        Fermi energy fails to converge.
    TypeError
        Electronic temperature is not given as `Tensor` or the derivative
        order is not an integer.
    ValueError
        Electronic temperature is not a scalar or negative, the number of
        electrons exceeds the number of orbitals, or the derivative order is
        negative.

    Note
    ----
    Derivatives (with respect to the orbital energies, and everything
    upstream of them, such as positions and fields, to `kt` and to `nel`):

    - ``k`` undamped Newton steps ``mu <- mu - g / g'`` of the number of
      electrons ``g`` from a converged, detached start reproduce all
      derivatives of the exact Fermi energy up to order ``2**k - 1``. This is
      the reason for ``k = ceil(log2(diff_order + 1))``. Proof: the Newton map
      ``N`` fulfils ``N(mu*) = mu*`` and ``N'(mu*) = 0``, i.e.,
      ``N(mu) - mu* = (mu - mu*)**2 h`` with a smooth ``h`` (roughly
      ``g'' / (2 g')``). With the start ``mu_0 = mu*(eps_0)``, the deviation
      ``mu_0 - mu*(eps)`` is ``O(d eps)``, and applying the identity ``k``
      times gives ``mu_k - mu* = O(d eps**(2**k))``. By the chain rule, the
      derivatives of the occupations agree up to the same order. The
      argument holds for any parameter of the Newton map, i.e., also for the
      number of electrons (``d mu / d N = 1 / g'``) and for mixed
      derivatives.
    - The derivative of the occupations of an active channel with respect to
      its number of electrons sums to one over the orbitals. Channels
      without electrons and completely filled channels have no derivative
      with respect to it (they are not optimized). At zero temperature, it
      is the derivative of the aufbau filling: one for the partially
      occupied orbital, and for integer electrons for the lowest empty one
      (the derivative for adding electrons).
    - In a gap (integer electrons), an additional electron is distributed
      according to the thermal weights, ``d f_i / d N = w_i / sum_j w_j``
      with ``w = f (1 - f)``, i.e., mostly to the HOMO and the LUMO, which
      are tails of the Fermi function. Three parts keep this ratio exact far
      beyond the threshold of the search: the residual of the Newton steps
      is evaluated without cancellation (holes below and electrons above the
      integer number of electrons, see `_partition`), the Fermi energy is
      polished in log space (its position in the gap sets the ratio), and
      the Fermi function has no cutoff.
    - The derivatives are exact up to the forward residual ``e_0`` of the
      Fermi energy: the ``n``-th derivative has an error of the order of
      ``e_0**(2**k - n)``. For ``diff_order = 2**k - 1``, the highest order
      has an error of order ``e_0`` (the error of a hand-written implicit
      derivative), and the lower orders are more accurate.
    - The default order 3 (two steps) covers forces, Hessians,
      polarizabilities, dipole derivatives (up to second order in the
      occupations) and the first hyperpolarizability and derivative of the
      polarizability (third order). Fourth-order properties need order 4 or
      more (three steps).
    - Order 0 (no step) omits the change of the Fermi energy entirely, which
      makes even forces wrong. Order 1 (one step) is exact for forces only.
    - The origin of the idea is one-step differentiation (Bolte, Pauwels and
      Vaiter, NeurIPS 2023, see the references of this module,
      [Bolte2023]_): a detached solve, differentiate only the last step(s) of
      a fast algorithm. Corollary 2 shows that one step gives the exact
      Jacobian if the Jacobian of the iteration map vanishes at the fixed
      point, and Corollary 3 bounds the error of one step by
      ``L_J ||x_{k-1} - x*||`` for quadratically convergent maps such as
      Newton's method. Both are first-order results. The statement for
      higher orders (``2**k - 1``) is the proof above and not part of the
      paper.
    - The steps must move the value. A straight-through form
      ``mu + (change - change.detach())`` differentiates the occupations at
      the start instead of the end of the steps and leaves an error of order
      ``e_0`` for every ``k``.
    - Completely filled channels (as many electrons as orbitals, e.g., He or
      H\\ :sup:`-`) have no finite Fermi energy. Their occupations are exactly
      one and constant, i.e., all derivatives vanish.
    - The derivative of the Fermi energy is dropped (the Fermi energy stays
      frozen) if the derivative of the number of electrons,
      ``g' = sum f (1 - f) / kT``, is below ``(tiny / eps)**(1 / 2**k)``, or
      if the step is longer than ``_MAX_DIFF_STEP_KT`` times kT (a safety
      net, not reached after the polish). Below the floor, the derivatives
      of order ``2**k`` with respect to `nel` would overflow, since the
      ``m``-th one grows like ``g'**(1 - m)`` in a gap. The missing terms for
      the orbital energies and `kt` are of the order of ``f (1 - f)``, i.e.,
      negligible. For `nel`, the derivative of that channel is zero instead
      of summing to one over the orbitals. With the Fermi energy in the
      middle of a gap and ``kT = 1e-3``, this happens for gaps of more than
      about 690, 355 and 180 kT for the orders 1, 3 and 7 in double
      precision and 87 and 55 kT for the orders 1 and 3 in single
      precision (measured). Up to order ``2**k`` (one beyond the exact
      ones), the derivatives stay finite (measured); higher orders may
      overflow in a gap, since their true values do.
    - The search reads its convergence flags on the host, i.e., the function
      has data-dependent control flow. Autograd and the transforms
      ``jacrev``, ``jacfwd``, ``hessian`` and ``jvp`` of ``torch.func`` work,
      ``vmap`` does not, and ``torch.compile`` breaks the graph at every
      convergence check.
    """
    # the temperature is a single value for all systems and channels
    if not isinstance(kt, Tensor):
        raise TypeError("Electronic temperature must be `Tensor`.")
    if kt.numel() != 1:
        raise ValueError(
            f"Electronic temperature must be a scalar (shape {kt.shape})."
        )
    # a CPU scalar would be mixed with the tensors on the device of `emo`
    kt = kt.reshape(()).to(emo.device)

    if diff_order is None:
        diff_order = defaults.FERMI_DIFF_ORDER
    steps = _diff_steps(diff_order)
    if diff_order > _MAX_SINGLE_DIFF_ORDER and emo.dtype != torch.double:
        warnings.warn(
            f"Derivatives of order {diff_order} of the Fermi occupations are "
            f"dominated by rounding errors in {emo.dtype}; use double "
            "precision.",
            stacklevel=2,
        )

    eps = torch.finfo(emo.dtype).eps
    if thr is None:
        # 1e-4 keeps a margin to the tolerance of the SCF (5e-4, float32)
        thr = min(eps**0.5, 1e5 * eps, 1e-4)
    thresh = any_to_tensor(thr, device=emo.device, dtype=emo.dtype)

    # `nel` ([b, 2]) gets a trailing dimension for the subtraction from the
    # sum over the orbitals ([b, 2, 1]); it keeps its graph
    target = nel.unsqueeze(-1)
    # existing orbitals, padding is never occupied ([b, 2, n])
    valid = None if mask is None else (mask != 0)

    # Channels without electrons (e.g., beta channel of a doublet) and
    # channels with zero or negative temperature are not optimized.
    kt_pos = kt > 0.0
    occupied = target > eps  # [b, 2, 1]
    # dummy value to not divide by zero (or a negative value) in the search,
    # which returns immediately without active channels
    safe_kt = torch.where(kt_pos, kt, 1.0)

    # Invalid input, only read together with the convergence flag: the number
    # of electrons of an active channel must fit into the existing orbitals.
    # number of orbitals of each channel (int or [b, 2, 1])
    norb = emo.shape[-1] if valid is None else valid.sum(-1, keepdim=True)
    tol = _integer_tol(emo.dtype)
    too_many = (target > norb + tol) & occupied & kt_pos

    # A completely filled channel has no finite Fermi energy: the occupations
    # are exactly one and constant. Searching for a root would only move the
    # Fermi energy by about kT per Newton step in the tail of the Fermi
    # function and give spurious derivatives.
    full = occupied & ((target - norb).abs() <= tol)
    active = occupied & kt_pos & ~full  # [b, 2, 1]

    # Flags that are read with the first convergence check of the search (a
    # single host synchronization): invalid input, which aborts the search,
    # and the choice between aufbau and Fermi filling.
    checks = torch.stack([kt < 0.0, torch.any(too_many)])
    flags = kt_pos.unsqueeze(0)

    # Fermi energy without gradient tracking ([b, 2, 1]): initial guess, then
    # iterations until the number of electrons is converged
    with torch.no_grad():
        emo_d, kt_d, target_d = emo.detach(), safe_kt.detach(), target.detach()
        n0, below, above = _partition(target_d, emo_d, valid)
        e_fermi = _initial_fermi_energy(target_d, emo_d, mask)
        e_fermi, (negative, too_many), (positive,) = _fermi_energy_search(
            target_d - n0,
            below,
            above,
            emo_d,
            kt_d,
            e_fermi,
            active,
            valid,
            thresh,
            maxiter,
            checks,
            flags,
        )

    if negative:
        raise ValueError(f"Electronic Temperature must be non-negative ({kt}).")
    if too_many:
        raise ValueError(
            f"Number of electrons ({nel}) exceeds the number of orbitals."
        )

    # Zero temperature: Fermi distribution becomes aufbau filling. It does not
    # depend on the orbital energies, which stay in the graph with a vanishing
    # derivative (`kt_pos` is False), i.e., differentiating with respect to
    # them gives zero instead of an error, as for a positive temperature.
    if not positive:
        aufbau = _aufbau_occupation(
            target, emo.shape[-1], valid, emo.dtype, emo.device
        )
        return torch.where(kt_pos, emo, aufbau)

    # Differentiable Newton steps from the converged Fermi energy. `rel` are
    # the orbital energies relative to the converged Fermi energy ([b, 2, n])
    # and `shift` the Fermi energy after the steps relative to the converged
    # one ([b, 2, 1]), i.e., `rel - shift` is the exact distance to the Fermi
    # energy, with the graph of the orbital energies and the electrons.
    rel, shift = _attach_implicit_derivative(
        e_fermi, emo, kt, target - n0, below, active, valid, steps
    )

    _, tail, _ = _fermi_distribution(rel, shift, kt, valid, below)

    # Tails below eps times the smallest weight of a channel that is still
    # differentiated (floor of the steps) carry no derivative, but values down
    # to 1e-308 make subnormal numbers in the products of the SCF (density,
    # potential), which are several times slower on CPUs. They are set to zero
    # (at least those below tiny / eps, e.g., without steps).
    info = torch.finfo(emo.dtype)
    flush = torch.clamp(
        _diff_floor(emo.dtype, steps) * kt.detach() * info.eps,
        min=info.tiny / info.eps,
    )
    tail = torch.where(tail < flush, 0.0, tail)
    fermi = torch.where(below, 1.0 - tail, tail)

    # channels without electrons (e.g., beta channel of an H atom) are not
    # occupied here; completely filled channels are
    full_occ = (
        torch.ones_like(fermi) if valid is None else valid.to(fermi.dtype)
    )
    return torch.where(full, full_occ, torch.where(active, fermi, 0.0))


def _fermi_distribution(
    emo: Tensor,
    e_fermi: Tensor,
    kt: Tensor,
    valid: Tensor | None,
    below: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Fermi function, its tail and its derivative with respect to the Fermi
    energy.

    With ``x = (emo - e_fermi) / kt``, the occupation is ``f = sigmoid(-x)``.
    The tail is the hole ``1 - f = sigmoid(x)`` of the orbitals below the
    integer number of electrons (see `_partition`) and the occupation
    ``f = sigmoid(-x)`` of the others, i.e., the small one in a gap. It is
    evaluated directly, so it (and all its derivatives) is accurate down to
    the smallest floats instead of being lost to rounding of ``1 - f`` (or
    to a cutoff). This is required for the derivatives with respect to the
    number of electrons in a gap, which are ratios of tails (see
    `get_fermi_occupation`). The logistic function neither overflows nor
    produces NaN.

    Parameters
    ----------
    emo : Tensor
        Orbital energies.
    e_fermi : Tensor
        Fermi energy with a trailing singleton dimension.
    kt : Tensor
        Electronic temperature in atomic units (must be positive).
    valid : Tensor | None
        Orbitals that exist (``False`` for padding). Padding is never
        occupied and has no tail.
    below : Tensor
        Orbitals below the integer number of electrons.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Occupation of each orbital, its tail and the derivative of the
        occupation with respect to the Fermi energy, ``f * (1 - f) / kt``.
    """
    # `emo` ([b, 2, n]) was expanded to the second dim (for alpha/beta
    # electrons) and `e_fermi` has a trailing singleton dim ([b, 2, 1]) for the
    # subtraction in that dim
    arg = (emo - e_fermi) / kt  # [b, 2, n]

    # only singly occupied here (the channels are separate): 0 <= f <= 1
    tail = torch.sigmoid(torch.where(below, arg, -arg))
    fermi = torch.where(below, 1.0 - tail, tail)
    dfermi = tail * (1.0 - tail) / kt

    # padding is never occupied and does not contribute to the derivative
    if valid is not None:
        fermi = torch.where(valid, fermi, 0.0)
        tail = torch.where(valid, tail, 0.0)
        dfermi = torch.where(valid, dfermi, 0.0)

    return fermi, tail, dfermi


def _partition(
    target: Tensor, emo: Tensor, valid: Tensor | None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Split the orbitals of each channel at the nearest integer number of
    electrons.

    The ``n0 = round(target)`` lowest orbitals (by energy, padding excluded)
    are "below", the others "above". With ``delta = target - n0``, the number
    of electrons is ``n0 - B + A``, where ``B`` is the number of holes below
    and ``A`` the number of electrons above. Both are sums of tails in a gap,
    i.e., the residual ``delta + B - A`` has no cancellation, unlike
    ``target - sum(f)``, which loses everything below ``n0 * eps``.

    Parameters
    ----------
    target : Tensor
        Number of electrons with a trailing singleton dimension (detached).
    emo : Tensor
        Orbital energies (detached).
    valid : Tensor | None
        Orbitals that exist (``False`` for padding).

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Integer number of electrons ``n0`` ([b, 2, 1]) and the masks of the
        orbitals below and above ([b, 2, n]).
    """
    # padding is sorted last and never below
    key = emo if valid is None else torch.where(valid, emo, torch.inf)
    rank = key.argsort(dim=-1).argsort(dim=-1)

    n0 = target.round()
    below = rank < n0
    above = ~below if valid is None else (~below & valid)
    return n0, below, above


def _newton_step(
    delta: Tensor,
    tail: Tensor,
    dfermi: Tensor,
    below: Tensor,
    floor: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Newton step of the Fermi energy for the number of electrons.

    Parameters
    ----------
    delta : Tensor
        Number of electrons relative to the integer of the partition
        (``target - n0``, see `_partition`) with a trailing singleton
        dimension.
    tail : Tensor
        Hole of the orbitals below the integer number of electrons and
        occupation of the others (see `_fermi_distribution`).
    dfermi : Tensor
        Derivative of the occupation with respect to the Fermi energy.
    below : Tensor
        Orbitals below the integer number of electrons.
    floor : float
        Smallest derivative of the number of electrons that is divided by.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor]
        Deviation from the target number of electrons, change of the Fermi
        energy and if the derivative is large enough to divide by it (the
        change is meaningless otherwise).
    """
    # target - sum(f) = delta + holes below - electrons above without
    # cancellation (see `_partition`), summed over the orbitals of each
    # channel: [b, 2, n] -> [b, 2, 1]
    resid = delta + torch.where(below, tail, -tail).sum(-1, keepdim=True)
    total_dfermi = dfermi.sum(-1, keepdim=True)

    # do not divide by a vanishing derivative (large gap): the change is
    # meaningless there and must not produce inf or NaN in the backward pass
    ok = total_dfermi > floor
    change = resid / torch.where(ok, total_dfermi, 1.0)
    return resid, change, ok


def _log_residual(
    emo: Tensor,
    e_fermi: Tensor,
    kt: Tensor,
    delta: Tensor,
    below: Tensor,
    above: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Residual of the number of electrons in log space and its derivative with
    respect to the Fermi energy.

    The root of ``h = log(A + max(-delta, 0)) - log(B + max(delta, 0))`` is
    the root of the number of electrons (``delta + B - A = 0``, see
    `_partition`). Unlike the number of electrons, which is flat in a gap
    (the tails are below any threshold), ``h`` has a slope of about
    ``2 / kt`` there, i.e., it determines the Fermi energy to machine
    precision anywhere. ``h`` is increasing in the Fermi energy.

    Parameters
    ----------
    emo : Tensor
        Orbital energies.
    e_fermi : Tensor
        Fermi energy with a trailing singleton dimension.
    kt : Tensor
        Electronic temperature in atomic units (positive).
    delta : Tensor
        Number of electrons relative to the integer of the partition.
    below : Tensor
        Orbitals below the integer number of electrons.
    above : Tensor
        Orbitals above the integer number of electrons.

    Returns
    -------
    tuple[Tensor, Tensor]
        Residual ``h`` and its derivative (infinite or NaN only for channels
        that are not optimized).
    """
    arg = (emo - e_fermi) / kt
    log_f = torch.nn.functional.logsigmoid(-arg)  # occupation
    log_h = torch.nn.functional.logsigmoid(arg)  # hole
    log_w = log_f + log_h  # f (1 - f)

    def lse(x: Tensor, mask: Tensor) -> Tensor:
        return torch.logsumexp(torch.where(mask, x, -torch.inf), -1, True)

    # log(0) = -inf, which `logaddexp` ignores
    log_a = torch.logaddexp(lse(log_f, above), (-delta).clamp(min=0.0).log())
    log_b = torch.logaddexp(lse(log_h, below), delta.clamp(min=0.0).log())

    resid = log_a - log_b
    deriv = (
        torch.exp(lse(log_w, above) - log_a)
        + torch.exp(lse(log_w, below) - log_b)
    ) / kt
    return resid, deriv


def _polish(
    emo: Tensor,
    e_fermi: Tensor,
    kt: Tensor,
    delta: Tensor,
    below: Tensor,
    above: Tensor,
    active: Tensor,
) -> Tensor:
    """
    Newton steps of the converged Fermi energy on the log-space residual.

    A converged number of electrons does not determine the Fermi energy in
    a gap, since the tails are below the threshold anywhere between the
    orbitals. The derivatives with respect to the number of electrons are
    ratios of the tails of the HOMO and the LUMO there, i.e., they depend on
    the position of the Fermi energy. The log-space residual (see
    `_log_residual`) is almost linear in a gap and quadratically convergent
    otherwise. A step is only taken if it reduces the residual, and the
    number of steps is fixed (no host synchronization).

    Parameters
    ----------
    emo : Tensor
        Orbital energies.
    e_fermi : Tensor
        Converged Fermi energy with a trailing singleton dimension.
    kt : Tensor
        Electronic temperature in atomic units (positive).
    delta : Tensor
        Number of electrons relative to the integer of the partition.
    below : Tensor
        Orbitals below the integer number of electrons.
    above : Tensor
        Orbitals above the integer number of electrons.
    active : Tensor
        Channels for which the Fermi energy is optimized (the others keep
        it, the residual is not finite there).

    Returns
    -------
    Tensor
        Fermi energy with a trailing singleton dimension.
    """
    resid, deriv = _log_residual(emo, e_fermi, kt, delta, below, above)
    for _ in range(_POLISH_STEPS):
        step = e_fermi - resid / torch.where(active, deriv, 1.0)
        new_resid, new_deriv = _log_residual(emo, step, kt, delta, below, above)

        better = active & torch.isfinite(new_resid)
        better = better & (new_resid.abs() < resid.abs())
        e_fermi = torch.where(better, step, e_fermi)
        resid = torch.where(better, new_resid, resid)
        deriv = torch.where(better, new_deriv, deriv)

    return e_fermi


def _is_converged(
    resid: Tensor, e_fermi: Tensor, lo: Tensor, hi: Tensor, thresh: Tensor
) -> Tensor:
    """
    Convergence of the number of electrons.

    `e_fermi` is the offset of the Fermi energy from the reference of the
    search (see `_fermi_energy_search`). Once the bracket is narrower than
    ``4 eps`` (absolute, or relative for offsets beyond one), it is no longer
    narrowed, which happens for tiny temperatures. Such entries are accepted
    if the deviation is at most 32 times the threshold (and 1e-6), i.e., in
    practice only in double precision.
    """
    eps = torch.finfo(resid.dtype).eps

    collapsed = hi - lo <= 4.0 * eps * torch.clamp(e_fermi.abs(), min=1.0)
    loose = resid.abs() <= torch.clamp(32.0 * thresh, max=1e-6)
    return (resid.abs() <= thresh) | (collapsed & loose)


def _initial_fermi_energy(
    target: Tensor, emo: Tensor, mask: Tensor | None
) -> Tensor:
    """
    Initial guess for the Fermi energy.

    For integer electrons, the midpoint of HOMO and LUMO is used. For
    fractional electrons, the energy of the partially occupied orbital is
    used.

    Parameters
    ----------
    target : Tensor
        Number of electrons with a trailing singleton dimension.
    emo : Tensor
        Orbital energies.
    mask : Tensor | None
        Mask for the existing orbitals (``0`` for padding).

    Returns
    -------
    Tensor
        Fermi energy with a trailing singleton dimension.
    """
    e_mid, homo = get_fermi_energy(target.squeeze(-1), emo, mask=mask)

    # `emo` ([b, 2, n]) was expanded to the second dim (for alpha/beta
    # electrons) and we need to add a dim to `e_fermi` for subtraction in that
    # dim
    e_mid = e_mid.unsqueeze(-1)  # [b, 2] -> [b, 2, 1]
    e_homo = torch.gather(emo, -1, homo)  # [b, 2, 1]

    # midpoint for integer electrons (a gap), energy of the partially occupied
    # orbital for fractional electrons (no gap between HOMO and LUMO)
    integer = (target - target.round()).abs() <= _integer_tol(emo.dtype)
    return torch.where(integer, e_mid, e_homo)


def _fermi_energy_search(
    delta: Tensor,
    below: Tensor,
    above: Tensor,
    emo: Tensor,
    kt: Tensor,
    e_fermi: Tensor,
    active: Tensor,
    valid: Tensor | None,
    thresh: Tensor,
    maxiter: int,
    checks: Tensor,
    flags: Tensor,
) -> tuple[tuple[Tensor, Tensor], list[bool], list[bool]]:
    """
    Find the Fermi energy for which the occupations sum up to `target`.

    Newton steps are only accepted if they stay within a bracket that is
    updated from the sign of the residual, since the Newton iteration for the
    Fermi function is not globally convergent (e.g., overshooting for
    fractional numbers of electrons or vanishing derivatives for a large
    HOMO-LUMO gap). Otherwise, the bracket is bisected. Entries that are
    converged are frozen, i.e., batched results equal the individual results.

    The host is only synchronized every ``_CHECK_EVERY`` iterations (and in
    the first and last one), since iterations after convergence do not change
    the result.

    The converged Fermi energy is polished in log space (see `_polish`):
    in a gap, the number of electrons is converged anywhere between the
    orbitals, but the derivatives of the occupations depend on the position
    of the Fermi energy.

    Parameters
    ----------
    delta : Tensor
        Number of electrons relative to the integer of the partition with a
        trailing singleton dimension (see `_partition`).
    below : Tensor
        Orbitals below the integer number of electrons.
    above : Tensor
        Orbitals above the integer number of electrons.
    emo : Tensor
        Orbital energies.
    kt : Tensor
        Electronic temperature in atomic units (positive).
    e_fermi : Tensor
        Initial guess for the Fermi energy.
    active : Tensor
        Channels for which the Fermi energy is optimized.
    valid : Tensor | None
        Orbitals that exist (``False`` for padding).
    thresh : Tensor
        Threshold for the deviation of the number of electrons.
    maxiter : int
        Maximum number of iterations.
    checks : Tensor
        Boolean flags for invalid input (negative temperature, too many
        electrons). They are read together with the convergence flag in the
        first synchronization, and the search is aborted if any is set.
    flags : Tensor
        Further boolean flags that are only read in the first
        synchronization, which saves a separate one.

    Returns
    -------
    tuple[tuple[Tensor, Tensor], list[bool], list[bool]]
        Fermi energy with a trailing singleton dimension as the sum of a
        reference and an offset (more precise than their float sum), the
        flags for invalid input and the further flags.

    Raises
    ------
    RuntimeError
        Fermi energy fails to converge.
    """
    # Bracket: the number of electrons is monotonic in the Fermi energy and
    # zero (all orbitals) far below (above) the lowest (highest) orbital.
    # (padding is not part of the spectrum and would only widen the bracket)
    if valid is None:
        emin, emax = emo.amin(-1, keepdim=True), emo.amax(-1, keepdim=True)
    else:
        inf = torch.full_like(emo, torch.inf)
        emin = torch.where(valid, emo, inf).amin(-1, keepdim=True)
        emax = torch.where(valid, emo, -inf).amax(-1, keepdim=True)
        # channels without orbitals are never active, keep them finite
        emin = torch.where(torch.isfinite(emin), emin, 0.0)
        emax = torch.where(torch.isfinite(emax), emax, 0.0)

    lo = emin - 60.0 * kt  # [b, 2, 1]
    hi = emax + 60.0 * kt  # [b, 2, 1]
    e_fermi, lo, hi, active = torch.broadcast_tensors(e_fermi, lo, hi, active)
    ref = torch.minimum(torch.maximum(e_fermi, lo), hi)

    # The Fermi energy is searched relative to the initial guess `ref`. In
    # single precision, one ulp of an absolute Fermi energy of a few Hartree
    # (5e-7) can change the number of electrons by more than the threshold
    # for (almost) degenerate orbitals at the Fermi energy, i.e., no
    # representable Fermi energy is converged. Relative to `ref`, the
    # resolution is that of the (small) offset, and the orbital energies
    # close to the Fermi energy are exact after the subtraction.
    emo, lo, hi = emo - ref, lo - ref, hi - ref
    e_fermi = torch.zeros_like(ref)

    # iterate the Fermi energy of each channel, `fermi` and `dfermi` are
    # [b, 2, n], everything else ([b, 2, 1]) is per channel; `maxiter` updates
    # are made and the result of the last one is checked, too
    resid, done = None, None
    read: list[bool] = []
    for it in range(maxiter + 1):
        _, tail, dfermi = _fermi_distribution(emo, e_fermi, kt, valid, below)
        resid, change, ok = _newton_step(
            delta, tail, dfermi, below, _sqrttiny(emo.dtype)
        )

        # channels that are not optimized (no electrons) are always done
        done = ~active | _is_converged(resid, e_fermi, lo, hi, thresh)

        # check if all channels of all systems are converged (host
        # synchronization) and if the input was invalid; the flags only
        # need to be read once
        if it % _CHECK_EVERY == 0 or it == maxiter:
            if it == 0:
                all_done, *read = _read_flags(
                    torch.cat([torch.all(done).unsqueeze(0), checks, flags])
                )
            else:
                all_done = bool(torch.all(done).item())

            invalid, further = read[: checks.numel()], read[checks.numel() :]
            if all_done or any(invalid):
                e_fermi = _polish(emo, e_fermi, kt, delta, below, above, active)
                return (ref, e_fermi), invalid, further

        if it == maxiter:
            break

        # too few (many) electrons: Fermi energy is above (below) the root
        lo = torch.where(resid > 0.0, e_fermi, lo)
        hi = torch.where(resid < 0.0, e_fermi, hi)

        # Newton step if it stays within the bracket, bisection otherwise
        newton = e_fermi + change
        accept = ok & (newton > lo) & (newton < hi)
        step = torch.where(accept, newton, 0.5 * (lo + hi))

        # converged channels are frozen, i.e., a batch gives the same result
        # as the individual systems
        e_fermi = torch.where(done, e_fermi, step)

    msg = "Fermi energy failed to converge"
    if resid is not None and done is not None:
        # report the entries (batch and channel index) that did not converge
        idx = (~done).squeeze(-1).nonzero()
        bad = [[int(i.item()) for i in row.unbind(0)] for row in idx.unbind(0)]
        worst = torch.where(done, 0.0, resid.abs()).max().item()
        limit = thresh.max().item()
        msg += (
            f" within {maxiter} iterations (entries {bad}, largest deviation "
            f"of the number of electrons {worst:.3e}, threshold {limit:.3e})"
        )
    raise RuntimeError(f"{msg}.")


def _attach_implicit_derivative(
    e_fermi: tuple[Tensor, Tensor],
    emo: Tensor,
    kt: Tensor,
    delta: Tensor,
    below: Tensor,
    active: Tensor,
    valid: Tensor | None,
    steps: int,
) -> tuple[Tensor, Tensor]:
    """
    Attach the derivatives of the converged Fermi energy to the graph.

    The converged Fermi energy is constant. Every step is an undamped Newton
    step of the number of electrons, ``mu <- mu - g(mu) / g'(mu)``, and the
    result of one step is the start of the next one (and the Fermi energy of
    the occupations). Since the Newton iteration converges quadratically,
    ``steps`` steps yield the correct derivatives up to order
    ``2**steps - 1`` (proof, orders per property and references in
    `get_fermi_occupation`). Without any step, even gradients miss the change
    of the Fermi energy.

    Note that the steps must move the value: a straight-through form such as
    ``mu + (change - change.detach())`` has the same derivatives of ``mu``,
    but the occupations would be differentiated at the start instead of the
    end of the steps, leaving an error of the order of the deviation of the
    start from the root for every ``k``.

    Parameters
    ----------
    e_fermi : tuple[Tensor, Tensor]
        Converged Fermi energy without graph as the sum of a reference and an
        offset (see `_fermi_energy_search`).
    emo : Tensor
        Orbital energies.
    kt : Tensor
        Electronic temperature in atomic units (positive).
    delta : Tensor
        Number of electrons relative to the integer of the partition with a
        trailing singleton dimension (see `_partition`). It carries the
        graph of the number of electrons.
    below : Tensor
        Orbitals below the integer number of electrons.
    active : Tensor
        Channels for which the Fermi energy is optimized. The others (no
        electrons) keep their Fermi energy, since a Newton step would only
        produce meaningless values there.
    valid : Tensor | None
        Orbitals that exist (``False`` for padding).
    steps : int
        Number of differentiable Newton steps.

    Returns
    -------
    tuple[Tensor, Tensor]
        Orbital energies relative to the converged Fermi energy and the
        Fermi energy after the Newton steps relative to the converged one
        (initially zero). Use them in place of the orbital energies and the
        Fermi energy in `_fermi_distribution`. The split keeps the steps,
        which are tiny compared to the ulp of the Fermi energy in single
        precision, from being lost when they are added to it.
    """
    # orbital energies ([b, 2, n]) relative to the converged Fermi energy
    # ([b, 2, 1], constant), which carries the graph of the orbital energies
    ref, offset = e_fermi
    emo = (emo - ref) - offset
    shift = torch.zeros_like(ref)  # Fermi energy relative to the converged

    floor = _diff_floor(emo.dtype, steps)
    for _ in range(steps):
        _, tail, dfermi = _fermi_distribution(emo, shift, kt, valid, below)
        _, change, ok = _newton_step(delta, tail, dfermi, below, floor)
        # keep the Fermi energy where the step is meaningless (see above)
        take = ok & active & (torch.abs(change) <= _MAX_DIFF_STEP_KT * kt)
        shift = shift + torch.where(take, change, 0.0)

    return emo, shift


def _aufbau_occupation(
    target: Tensor,
    norb: int,
    valid: Tensor | None,
    dtype: torch.dtype,
    device: torch.device | None,
) -> Tensor:
    """
    Aufbau filling for (fractional) electrons.

    Padded orbitals are skipped, i.e., the electrons fill the existing
    orbitals in the given order.

    Parameters
    ----------
    target : Tensor
        Number of electrons with a trailing singleton dimension.
    norb : int
        Number of orbitals (including padding).
    valid : Tensor | None
        Orbitals that exist (``False`` for padding).
    dtype : torch.dtype
        Data type of the occupation.
    device : torch.device | None
        Device of the occupation.

    Returns
    -------
    Tensor
        Occupation numbers with the last dimension of size `norb`.
    """
    # The orbital with the (0-based) position `i` holds `nel - i` electrons,
    # limited to [0, 1]: whole electrons fill the lower orbitals (1), the
    # remainder of fractional electrons the next one and all higher orbitals
    # stay empty (0). `target` is [b, 2, 1] and the positions are [norb] (or
    # [b, 1, norb] with padding), i.e., the result is [b, 2, norb].
    if valid is None:
        idxs = torch.arange(norb, device=device, dtype=dtype)
        return _fill(target - idxs)

    # position of each orbital among the existing orbitals: padding does not
    # count, i.e., the electrons continue after it
    rank = torch.cumsum(valid.to(dtype), dim=-1) - 1.0
    return torch.where(valid, _fill(target - rank), 0.0)


def _fill(x: Tensor) -> Tensor:
    """
    Occupation ``min(max(x, 0), 1)`` of an orbital that `x` electrons reach.

    Unlike `torch.clamp`, whose derivative vanishes at both bounds, the
    derivative with respect to the electrons is one for ``0 <= x < 1``, i.e.,
    for integer electrons, it belongs to the lowest empty orbital (adding
    electrons) and the derivatives of all orbitals sum to one.
    """
    return torch.where(x >= 1.0, 1.0, torch.where(x < 0.0, 0.0, x))
