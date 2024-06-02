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

Parts of the Fermi smearing are taken from https://github.com/tbmalt/tbmalt
"""

from __future__ import annotations

import torch

from dxtb._src.typing import DD, Tensor

from ..constants import defaults

__all__ = [
    "get_alpha_beta_occupation",
    "get_aufbau_occupation",
    "get_fermi_energy",
    "get_fermi_occupation",
]


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
        Number of unpaired electrons. If ``None``, spin is figured out automatically.

    Returns
    -------
    Tensor
        Alpha (first column, 0 index) and beta (second column, 1 index) electrons.

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

    nuhf = torch.where(
        torch.remainder(uhf, 2) == torch.remainder(nel.round(), 2),
        uhf,
        torch.remainder(nel.round(), 2),
    )

    diff = torch.minimum(nuhf, nel)
    nb = (nel - diff) / 2.0
    na = nb + diff

    return torch.stack([na, nb], dim=-1)


def get_aufbau_occupation(norb: Tensor, nel: Tensor) -> Tensor:
    """
    Set occupation numbers according to the aufbau principle.
    The number of electrons is a real number and can be fractional.

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

    # We represent the aufbau filling with a heaviside function, using the following steps
    # 1. creating orbital indices using arange from 1 to norb, inclusively
    idxs = torch.arange(1, 1 + torch.max(norb).item(), device=nel.device)
    occupation = torch.heaviside(
        # 2. remove the orbital index from the total number of electrons
        #    (negative numbers are filled with ones, positive numbers with zeros)
        # 3. fractional occupation will be in the range [-1, 0], therefore we round up
        torch.ceil(nel.unsqueeze(-1) - idxs.unsqueeze(-2)),
        # 4. heaviside uses the actual values at 0, therefore we provide the remainder
        # 5. to not lose whole electrons we take the negative and add one
        torch.remainder(nel, -1).unsqueeze(-1) + 1,
    )

    return occupation.flatten() if nel.dim() == 0 else occupation


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
        Number of electrons.
    emo : Tensor
        Orbital energies
    mask : Tensor | None, optional
        Mask from orbitals to avoid reading padding as LUMO for elements
        without LUMO due to minimal basis.

    Returns
    -------
    tuple[Tensor, Tensor]
        Fermi energy and index of HOMO.
    """
    zero = torch.tensor(0.0, device=emo.device, dtype=emo.dtype)

    occ = torch.ones_like(emo)
    occ_cs = occ.cumsum(-1) - nel.unsqueeze(-1)

    # transition: negative values indicate end of occupied orbitals
    temp = occ_cs >= (-torch.finfo(emo.dtype).resolution * 5)

    # index of first non-negative value and unsqueeze for stacking;
    # stacking will happen along that dim
    homo = torch.argmax(temp.type(torch.long), dim=-1).unsqueeze(-1)

    # some atoms (e.g., He) do not have a LUMO because of the valence basis and
    # the LUMO index becomes larger than No. MOs
    lumo_missing = occ.sum(-1, keepdim=True) - 1 <= homo
    gap = torch.where(
        lumo_missing,
        torch.cat((homo, homo), -1),  # Fermi energy becomes HOMO energy
        torch.cat((homo, homo + 1), -1),
    )

    # Fermi energy as midpoint between HOMO and LUMO
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
    kt: Tensor | None = None,
    mask: Tensor | None = None,
    thr: dict[torch.dtype, Tensor] | None = None,
    maxiter: int = 200,
) -> Tensor:
    """
    Set occupation numbers according to Fermi distribution.

    The orbital energies `emo` must already have the correct shape for using
    alpha/beta electron channels. Spreading to the channels can be done with
    `emo.unsqueeze(-2).expand([*nel.shape, -1])`.

    Parameters
    ----------
    nel : Tensor
        Number of electrons.
    emo : Tensor
        Orbital energies.
    kt : Tensor | None, optional
        Electronic temperature in atomic units, by default None.
    mask : Tensor | None, optional
        Mask for Fermi energy. Just passed through.
    thr : Tensor | None, optional
        Threshold for converging Fermi energy, by default None.
    maxiter : int, optional
        Maximum number of iterations for converging Fermi energy, by default 200.

    Returns
    -------
    Tensor
        Occupation numbers.

    Raises
    ------
    RuntimeError
        Fermi energy fails to converge.
    TypeError
        Electronic temperature is not given as `Tensor`.
    ValueError
        Electronic temperature is negative or number of electrons is zero.
    """

    # wrong type of kt
    if not isinstance(kt, Tensor) and kt is not None:
        raise TypeError("Electronic temperature must be `Tensor` or ``None``.")

    # negative etemp
    if kt is not None and torch.any(kt < 0.0):
        raise ValueError(f"Electronic Temperature must be positive or None ({kt}).")

    dd: DD = {"device": emo.device, "dtype": emo.dtype}
    eps = torch.tensor(torch.finfo(emo.dtype).eps, **dd)
    zero = torch.tensor(0.0, **dd)

    # no valence electrons
    if (torch.abs(nel.sum(-1)) < eps).any():
        return torch.zeros_like(emo)
        raise ValueError("Number of valence electrons cannot be zero.")

    if thr is None:
        thr = defaults.FERMI_THRESH
    thresh = thr.get(emo.dtype, torch.tensor(1e-5, dtype=torch.float)).to(emo.device)

    e_fermi, homo = get_fermi_energy(nel, emo, mask=mask)

    # `emo` ([b, 2, n]) was expanded to second dim (for alpha/beta electrons)
    # and we need to add a dim to `e_fermi` for subtraction in that dim
    e_fermi = e_fermi.view([*nel.shape, -1])  # [b, 2, 1]

    # check if (beta) channel contains electrons
    not_empty = nel.unsqueeze(-1) != 0
    emo = torch.where(not_empty, emo, zero)

    # iterate fermi energy
    for _ in range(maxiter):
        exponent = (emo - e_fermi) / kt
        eterm = torch.exp(torch.where(exponent < 50, exponent, zero))

        # only singly occupied here         v
        fermi = torch.where(exponent < 50, 1.0 / (eterm + 1.0), zero)
        dfermi = torch.where(exponent < 50, eterm / (kt * (eterm + 1.0) ** 2), eps)

        _nel = torch.sum(fermi, dim=-1, keepdim=True)
        change = (homo - _nel + 1) / torch.sum(dfermi, dim=-1, keepdim=True)
        e_fermi += change

        if torch.all(torch.abs(homo - _nel + 1) <= thresh):
            # check if beta channel is empty
            return torch.where(not_empty, fermi, torch.zeros_like(fermi))

    raise RuntimeError("Fermi energy failed to converge.")
