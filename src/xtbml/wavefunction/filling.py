"""
Handle the occupation of the orbitals with electrons.
"""

import torch
from ..typing import Tensor
from ..utils import t2int


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


def get_fermi_occupation(
    nel: Tensor, occupation: Tensor, emo: Tensor, kt: Tensor, maxiter: int, thr
):

    kt = kt * 100

    # indexing starts with zero -> -1
    homo = t2int(torch.floor(nel / 2))
    homo = homo + 1 if nel / 2 % 1 > 0.5 else homo
    occt = homo

    e_fermi = 0.5 * (emo[max(homo - 1, 0)] + emo[min(homo, emo.size(0) - 1)])

    occ = torch.zeros_like(occupation)
    eps = torch.tensor(torch.finfo(emo.dtype).eps, device=emo.device)
    zero = emo.new_tensor(0.0)

    for _ in range(maxiter):
        exponent = (emo - e_fermi) / kt
        eterm = torch.exp(exponent)

        fermi = torch.where(exponent < 50, 1.0 / (eterm + 1.0), zero)
        dfermi = torch.where(exponent < 50, eterm / (kt * (eterm + 1.0) ** 2), eps)

        n_el = fermi.sum(-1)
        change = (occt - n_el) / (dfermi.sum(-1))
        e_fermi += change

        if abs(occt - n_el) <= thr:
            break

    print(fermi)
    return occ * 2
