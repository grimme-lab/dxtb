"""
Handle the occupation of the orbitals with electrons.
"""

# NOTE: Parts of the Fermi smearing are taken from https://github.com/tbmalt/tbmalt


from __future__ import annotations

import torch

from ..typing import Tensor
from ..constants import defaults


def get_aufbau_occupation(
    norb: Tensor,
    nel: Tensor,
) -> Tensor:
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
    >>> get_aufbau_occupation(torch.tensor(5), torch.tensor(1.))  # 1 el. in 5 orb.
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
    """

    # We represent the aufbau filling with a heaviside function, using the following steps
    occupation = torch.heaviside(
        # 1. creating orbital indices using arange from 1 to norb, inclusively
        # 2. remove the orbital index from the total number of electrons
        #    (negative numbers are filled with ones, positive numbers with zeros)
        # 3. fractional occupations will be in the range [-1, 0], therefore we round up
        torch.ceil(
            nel.unsqueeze(-1)
            - torch.arange(1, 1 + torch.max(norb).item()).unsqueeze(-2)
        ),
        # 4. heaviside uses the actual values at 0, therefore we provide the remainder
        # 5. to not lose whole electrons we take the negative and add one
        torch.remainder(nel, -1).unsqueeze(-1) + 1,
    )

    return occupation.flatten() if nel.dim() == 0 else occupation


def get_fermi_energy(
    nel: Tensor, emo: Tensor, max_orb_occ: float = 2.0
) -> tuple[Tensor, Tensor]:
    """
    Get Fermi energy as midpoint between the HOMO and LUMO.

    Parameters
    ----------
    nel : Tensor
        Number of electrons.
    emo : Tensor
        Orbital energies
    max_orb_occ : float, optional
        Maximum orbital occupation, by default 2.0

    Returns
    -------
    tuple[Tensor, Tensor]
        Fermi energy and index of HOMO.
    """
    shp = torch.Size([*emo.shape[:-1], -1])

    # maximum occupation of each orbital
    occ = torch.ones_like(emo) * max_orb_occ

    # transition: negative values indicate end of occupied orbitals
    if emo.ndim == 1:
        occ_cs = occ.cumsum(-1) - nel
    else:
        # transpose sum because `nel` may contain multiple values
        occ_cs = occ.cumsum(-1).T - nel

    temp = occ_cs >= (-torch.finfo(emo.dtype).resolution * 5)
    if emo.ndim > 1:
        temp = temp.T

    # index of first non-negative value
    homo = torch.argmax(temp.type(torch.long), dim=-1).view(shp)

    # some atoms (e.g., He) do not have a LUMO because of the valence basis
    hl = torch.where(
        homo + 1 >= emo.size(-1),  # LUMO index larger than number of MOs
        torch.cat((homo, homo), -1),
        torch.cat((homo, homo + 1), -1),
    )
    # FIXME: BATCHED CALCULATIONS
    # In batched calculations the missing LUMO is replaced by padding, which is
    # not caught by the above `torch.where`. Consequently, the Fermi energy is
    # no correct. To fix this, one would require a mask from `numbers`. The
    # `numbers` are, however, currently not passed down to the SCF.

    e_fermi = torch.gather(emo, -1, hl).mean(-1).view(shp)
    return e_fermi, homo


def get_fermi_occupation(
    nel: Tensor,
    emo: Tensor,
    kt: Tensor | None = None,
    thr: dict[torch.dtype, Tensor] | None = None,
    maxiter: int = 200,
    max_orb_occ: float = 2.0,
) -> Tensor:
    """
    Set occupation numbers according to Fermi distribution.

    Parameters
    ----------
    nel : Tensor
        Number of electrons.
    emo : Tensor
        Orbital energies
    kt : Tensor | None, optional
        Electronic temperature in atomic units, by default None.
    thr : Tensor | None, optional
        Threshold for converging Fermi energy, by default None.
    maxiter : int, optional
        Maximum number of iterations for converging Fermi energy, by default 200.
    max_orb_occ : float, optional
        Maximum orbital occupation, by default 2.0.

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
        raise TypeError("Electronic temperature must be `Tensor` or `None`.")

    # negative etemp
    if kt is not None and (kt < 0.0).any():
        raise ValueError(f"Electronic Temperature must be positive or None ({kt}).")

    eps = emo.new_tensor(torch.finfo(emo.dtype).eps)
    zero = emo.new_tensor(0.0)

    # no electrons
    if (torch.abs(nel) < eps).any():
        raise ValueError("Number of elections cannot be zero.")

    # no electronic temperature: just return aufbau occupation
    if kt is None or torch.all(kt < 3e-7):  # 0.1 Kelvin * K2AU
        return 2.0 * get_aufbau_occupation(emo.new_tensor(emo.shape[-1]), nel / 2.0)

    if thr is None:
        thr = defaults.THRESH
    thresh = thr.get(emo.dtype, torch.tensor(1e-5, dtype=torch.float)).to(emo.device)

    # iterate fermi energy
    e_fermi, homo = get_fermi_energy(nel, emo)
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
            return max_orb_occ * fermi

    raise RuntimeError("Fermi energy failed to converge.")
