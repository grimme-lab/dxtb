"""
Bond order estimation
---------------------

Module for estimating or guessing bond orders between atoms. This module uses
a geometric model to obtain the optimal bond distance between a pair of atoms,
which is compared with the actual distance between the atoms to obtain the
liklihood of the bond being present.

Based on S. Spicher and S. Grimme, *Angew. Chem. Int. Ed.*, **2020**, 59, 15665–15673
(`DOI <https://doi.org10.1002/anie.202004239>`__).

Example
-------
>>> import torch
>>> from xtbml.bond import guess_bond_order
>>> numbers = torch.tensor([7, 7, 1, 1, 1, 1, 1, 1])
>>> positions = torch.tensor([
...     [-2.98334550857544, -0.08808205276728, +0.00000000000000],
...     [+2.98334550857544, +0.08808205276728, +0.00000000000000],
...     [-4.07920360565186, +0.25775116682053, +1.52985656261444],
...     [-1.60526800155640, +1.24380481243134, +0.00000000000000],
...     [-4.07920360565186, +0.25775116682053, -1.52985656261444],
...     [+4.07920360565186, -0.25775116682053, -1.52985656261444],
...     [+1.60526800155640, -1.24380481243134, +0.00000000000000],
...     [+4.07920360565186, -0.25775116682053, +1.52985656261444],
... ])
>>> cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
>>> bond_order = guess_bond_order(numbers, positions, cn)
>>> print(bond_order)
tensor([[0.0000, 0.0000, 0.4403, 0.4334, 0.4403, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4403, 0.4334, 0.4403],
        [0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4334, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4334, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.4403, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
>>> print(bond_order > 0.3)
tensor([[False, False,  True,  True,  True, False, False, False],
        [False, False, False, False, False,  True,  True,  True],
        [ True, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False],
        [ True, False, False, False, False, False, False, False],
        [False,  True, False, False, False, False, False, False],
        [False,  True, False, False, False, False, False, False],
        [False,  True, False, False, False, False, False, False]])

"""

import torch
from typing import Callable, Any

from .ncoord.ncoord import erf_count

Tensor = torch.Tensor


_en = torch.tensor(
    [
        *[+0.00000000],
        *[+2.30085633, +2.78445145, +1.52956084, +1.51714704, +2.20568300],
        *[+2.49640820, +2.81007174, +4.51078438, +4.67476223, +3.29383610],
        *[+2.84505365, +2.20047950, +2.31739628, +2.03636974, +1.97558064],
        *[+2.13446570, +2.91638164, +1.54098156, +2.91656301, +2.26312147],
        *[+2.25621439, +1.32628677, +2.27050569, +1.86790977, +2.44759456],
        *[+2.49480042, +2.91545568, +3.25897750, +2.68723778, +1.86132251],
        *[+2.01200832, +1.97030722, +1.95495427, +2.68920990, +2.84503857],
        *[+2.61591858, +2.64188286, +2.28442252, +1.33011187, +1.19809388],
        *[+1.89181390, +2.40186898, +1.89282464, +3.09963488, +2.50677823],
        *[+2.61196704, +2.09943450, +2.66930105, +1.78349472, +2.09634533],
        *[+2.00028974, +1.99869908, +2.59072029, +2.54497829, +2.52387890],
        *[+2.30204667, +1.60119300, +2.00000000, +2.00000000, +2.00000000],
        *[+2.00000000, +2.00000000, +2.00000000, +2.00000000, +2.00000000],
        *[+2.00000000, +2.00000000, +2.00000000, +2.00000000, +2.00000000],
        *[+2.00000000, +2.30089349, +1.75039077, +1.51785130, +2.62972945],
        *[+2.75372921, +2.62540906, +2.55860939, +3.32492356, +2.65140898],
        *[+1.52014458, +2.54984804, +1.72021963, +2.69303422, +1.81031095],
        *[+2.34224386],
    ]
)
"""
Electronegativity parameter used to determine polarity of bonds.
"""

_r0 = torch.tensor(
    [
        *[+0.00000000],
        *[+0.55682207, +0.80966997, +2.49092101, +1.91705642, +1.35974851],
        *[+0.98310699, +0.98423007, +0.76716063, +1.06139799, +1.17736822],
        *[+2.85570926, +2.56149012, +2.31673425, +2.03181740, +1.82568535],
        *[+1.73685958, +1.97498207, +2.00136196, +3.58772537, +2.68096221],
        *[+2.23355957, +2.33135502, +2.15870365, +2.10522128, +2.16376162],
        *[+2.10804037, +1.96460045, +2.00476257, +2.22628712, +2.43846700],
        *[+2.39408483, +2.24245792, +2.05751204, +2.15427677, +2.27191920],
        *[+2.19722638, +3.80910350, +3.26020971, +2.99716916, +2.71707818],
        *[+2.34950167, +2.11644818, +2.47180659, +2.32198800, +2.32809515],
        *[+2.15244869, +2.55958313, +2.59141300, +2.62030465, +2.39935278],
        *[+2.56912355, +2.54374096, +2.56914830, +2.53680807, +4.24537037],
        *[+3.66542289, +3.19903011, +2.80000000, +2.80000000, +2.80000000],
        *[+2.80000000, +2.80000000, +2.80000000, +2.80000000, +2.80000000],
        *[+2.80000000, +2.80000000, +2.80000000, +2.80000000, +2.80000000],
        *[+2.80000000, +2.34880037, +2.37597108, +2.49067697, +2.14100577],
        *[+2.33473532, +2.19498900, +2.12678348, +2.34895048, +2.33422774],
        *[+2.86560827, +2.62488837, +2.88376127, +2.75174124, +2.83054552],
        *[+2.63264944],
    ]
)
"""
Van-der-Waals radii for each element.
"""

_cf = torch.tensor(
    [
        *[+0.00000000],
        *[+0.17957827, +0.25584045, -0.02485871, +0.00374217, +0.05646607],
        *[+0.10514203, +0.09753494, +0.30470380, +0.23261783, +0.36752208],
        *[+0.00131819, -0.00368122, -0.01364510, +0.04265789, +0.07583916],
        *[+0.08973207, -0.00589677, +0.13689929, -0.01861307, +0.11061699],
        *[+0.10201137, +0.05426229, +0.06014681, +0.05667719, +0.02992924],
        *[+0.03764312, +0.06140790, +0.08563465, +0.03707679, +0.03053526],
        *[-0.00843454, +0.01887497, +0.06876354, +0.01370795, -0.01129196],
        *[+0.07226529, +0.01005367, +0.01541506, +0.05301365, +0.07066571],
        *[+0.07637611, +0.07873977, +0.02997732, +0.04745400, +0.04582912],
        *[+0.10557321, +0.02167468, +0.05463616, +0.05370913, +0.05985441],
        *[+0.02793994, +0.02922983, +0.02220438, +0.03340460, -0.04110969],
        *[-0.01987240, +0.07260201, +0.07700000, +0.07700000, +0.07700000],
        *[+0.07700000, +0.07700000, +0.07700000, +0.07700000, +0.07700000],
        *[+0.07700000, +0.07700000, +0.07700000, +0.07700000, +0.07700000],
        *[+0.07700000, +0.08379100, +0.07314553, +0.05318438, +0.06799334],
        *[+0.04671159, +0.06758819, +0.09488437, +0.07556405, +0.13384502],
        *[+0.03203572, +0.04235009, +0.03153769, -0.00152488, +0.02714675],
        *[+0.04800662],
    ]
)
"""
Coordination number based scaling factor.
"""

_ir = torch.tensor([0] + 2 * [1] + 8 * [2] + 8 * [3] + 18 * [4] + 18 * [5] + 32 * [6])
"""
Row index in the periodic table
"""

_p1 = (
    0.01
    * torch.tensor(
        [0.0, 29.84522887, -1.70549806, 6.54013762, 6.39169003, 6.00, 5.6],
    )[_ir]
)
"""
Polynomial parameters for contributions linear in the electronegativity difference.
"""

_p2 = (
    0.01
    * torch.tensor(
        [0.0, -8.87843763, 2.10878369, 0.08009374, -0.85808076, -1.15, -1.3],
    )[_ir]
)
"""
Polynomial parameters for contributions quadratic in the electronegativity difference.
"""


def guess_bond_length(
    numbers: Tensor,
    cn: Tensor,
) -> Tensor:
    """
    Estimate equilibrium bond lengths using a geometric model.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system
    cn : Tensor
        Coordination numbers for all atoms in the system

    Returns
    -------
    Tensor
        Estimated bond lengths for all atom pairs


    Example
    -------
    >>> import torch
    >>> from xtbml.bond import guess_bond_length
    >>> numbers = torch.tensor([6, 8, 7, 1, 1, 1])
    >>> cn = torch.tensor([3.0059586, 1.0318390, 3.0268824, 1.0061584, 1.0036336, 0.9989871])
    >>> print(guess_bond_length(numbers, cn))
    tensor([[2.5983, 2.2588, 2.5871, 1.9833, 1.9828, 1.9820],
            [2.2588, 2.1631, 2.2855, 1.5542, 1.5538, 1.5531],
            [2.5871, 2.2855, 2.5589, 1.8902, 1.8897, 1.8890],
            [1.9833, 1.5542, 1.8902, 1.4750, 1.4746, 1.4737],
            [1.9828, 1.5538, 1.8897, 1.4746, 1.4741, 1.4733],
            [1.9820, 1.5531, 1.8890, 1.4737, 1.4733, 1.4724]])
    """

    r0 = _r0[numbers].type(cn.dtype)
    en = _en[numbers].type(cn.dtype)
    cf = _cf[numbers].type(cn.dtype)
    p1 = _p1[numbers].type(cn.dtype)
    p2 = _p2[numbers].type(cn.dtype)

    ratom = r0 + cf * cn
    ediff = torch.abs(en.unsqueeze(-1) - en.unsqueeze(-2))
    scale = (
        ediff.new_ones(ediff.shape)
        - (p1.unsqueeze(-1) + p1.unsqueeze(-2)) / 2 * ediff
        - (p2.unsqueeze(-1) + p2.unsqueeze(-2)) / 2 * ediff**2
    )

    return scale * (ratom.unsqueeze(-1) + ratom.unsqueeze(-2))


def guess_bond_order(
    numbers: Tensor,
    positions: Tensor,
    cn: Tensor,
    counting_function: Callable[[Tensor, Tensor, Any], Tensor] = erf_count,
    **kwargs,
) -> Tensor:
    """
    Try to guess whether an atom pair is bonded using a geometric criterium.
    This measure is based on model taking into account the coordination number
    of each atom as well as the polarity of the bond.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers for all atoms in the system
    positions : Tensor
        Cartesian coordinates for all atoms in the system
    cn : Tensor
        Coordination numbers for all atoms in the system
    counting_function : callable
        Function to determine whether two atoms are bonded,
        additional arguments are passed to the counting function

    Returns
    -------
    Tensor
        Bond order for all atom pairs

    Example
    -------
    >>> import torch
    >>> from xtbml.bond import guess_bond_order
    >>> from xtbml.exlibs.tbmalt import batch
    >>> numbers = batch.pack((
    ...     torch.tensor([7, 1, 1, 1]),
    ...     torch.tensor([6, 8, 8, 1, 1]),
    ... ))
    >>> positions = batch.pack((
    ...     torch.tensor([
    ...         [+0.00000000000000, +0.00000000000000, -0.54524837997150],
    ...         [-0.88451840382282, +1.53203081565085, +0.18174945999050],
    ...         [-0.88451840382282, -1.53203081565085, +0.18174945999050],
    ...         [+1.76903680764564, +0.00000000000000, +0.18174945999050],
    ...     ]),
    ...     torch.tensor([
    ...         [-0.53424386915034, -0.55717948166537, +0.00000000000000],
    ...         [+0.21336223456096, +1.81136801357186, +0.00000000000000],
    ...         [+0.82345103924195, -2.42214694643037, +0.00000000000000],
    ...         [-2.59516465056138, -0.70672678063558, +0.00000000000000],
    ...         [+2.09259524590881, +1.87468519515944, +0.00000000000000],
    ...     ]),
    ... ))
    >>> cn = torch.tensor([
    ...     [2.9901006, 0.9977214, 0.9977214, 0.9977214, 0.0000000],
    ...     [3.0093639, 2.0046251, 1.0187057, 0.9978270, 1.0069743],
    ... ])
    >>> bond_order = guess_bond_order(numbers, positions, cn)
    >>> print(bond_order[0, ...])
    tensor([[0.0000, 0.4392, 0.4392, 0.4392, 0.0000],
            [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.4392, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
    >>> print(bond_order[1, ...])
    tensor([[0.0000, 0.5935, 0.4043, 0.3262, 0.0000],
            [0.5935, 0.0000, 0.0000, 0.0000, 0.3347],
            [0.4043, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.3262, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.3347, 0.0000, 0.0000, 0.0000]])
    """

    eps = torch.tensor(torch.finfo(positions.dtype).eps, dtype=positions.dtype)
    real = numbers != 0
    mask = real.unsqueeze(-2) * real.unsqueeze(-1)
    mask.diagonal(dim1=-2, dim2=-1).fill_(False)
    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2),
        eps,
    )

    bond_length = guess_bond_length(numbers, cn)
    return torch.where(
        mask,
        counting_function(distances, bond_length, **kwargs),
        torch.tensor(0.0, dtype=distances.dtype),
    )
