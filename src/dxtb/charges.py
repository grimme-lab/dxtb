"""
Electronegativity equilibration charge model
============================================

Implementation of the electronegativity equlibration model for obtaining
atomic partial charges as well as atom-resolved electrostatic energies.

Example
-------
>>> import torch
>>> import xtbml.charges as charges
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
>>> total_charge = torch.tensor(0.0)
>>> cn = torch.tensor([3.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
>>> eeq = charges.ChargeModel.param2019()
>>> energy, qat = charges.solve(numbers, positions, total_charge, eeq, cn)
>>> print(torch.sum(energy, -1))
tensor(-0.1750)
>>> print(qat)
tensor([-0.8347, -0.8347,  0.2731,  0.2886,  0.2731,  0.2731,  0.2886,  0.2731])
"""

import math

import torch

from .typing import Tensor, TensorLike
from .utils import real_atoms, real_pairs


class ChargeModel(TensorLike):
    """
    Model for electronegativity equilibration
    """

    chi: Tensor
    """Electronegativity for each element"""

    kcn: Tensor
    """Coordination number dependency of the electronegativity"""

    eta: Tensor
    """Chemical hardness for each element"""

    rad: Tensor
    """Atomic radii for each element"""

    __slots__ = ["chi", "kcn", "eta", "rad"]

    def __init__(
        self,
        chi: Tensor,
        kcn: Tensor,
        eta: Tensor,
        rad: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(device, dtype)
        self.chi = chi
        self.kcn = kcn
        self.eta = eta
        self.rad = rad

        if any(
            tensor.device != self.device
            for tensor in (self.chi, self.kcn, self.eta, self.rad)
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any(
            tensor.dtype != self.dtype
            for tensor in (self.chi, self.kcn, self.eta, self.rad)
        ):
            raise RuntimeError("All tensors must have the same dtype!")

    @classmethod
    def param2019(cls) -> "ChargeModel":
        """
        Electronegativity equilibration charge model published in

        - E. Caldeweyher, S. Ehlert, A. Hansen, H. Neugebauer, S. Spicher, C. Bannwarth
          and S. Grimme, *J. Chem. Phys.*, **2019**, 150, 154122.
          DOI: `10.1063/1.5090222 <https://dx.doi.org/10.1063/1.5090222>`__
        """

        return cls(
            _chi2019,
            _kcn2019,
            _eta2019,
            _rad2019,
        )

    def to(self, device: torch.device) -> "ChargeModel":
        """
        Returns a copy of the `ChargeModel` instance on the specified device.

        This method creates and returns a new copy of the `ChargeModel` instance
        on the specified device "``device``".

        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.

        Returns
        -------
        ChargeModel
            A copy of the `ChargeModel` instance placed on the specified device.

        Notes
        -----
        If the `ChargeModel` instance is already on the desired device `self`
        will be returned.
        """
        if self.device == device:
            return self

        return self.__class__(
            self.chi.to(device=device),
            self.kcn.to(device=device),
            self.eta.to(device=device),
            self.rad.to(device=device),
            device=device,
        )

    def type(self, dtype: torch.dtype) -> "ChargeModel":
        """
        Returns a copy of the `ChargeModel` instance with specified floating point type.
        This method creates and returns a new copy of the `ChargeModel` instance
        with the specified dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Type of the

        Returns
        -------
        ChargeModel
            A copy of the `ChargeModel` instance with the specified dtype.

        Notes
        -----
        If the `ChargeModel` instance has already the desired dtype `self` will
        be returned.
        """
        if self.dtype == dtype:
            return self

        return self.__class__(
            self.chi.type(dtype),
            self.kcn.type(dtype),
            self.eta.type(dtype),
            self.rad.type(dtype),
            dtype=dtype,
        )


_chi2019 = torch.tensor(
    [
        *[+0.00000000],
        *[+1.23695041, +1.26590957, +0.54341808, +0.99666991, +1.26691604],
        *[+1.40028282, +1.55819364, +1.56866440, +1.57540015, +1.15056627],
        *[+0.55936220, +0.72373742, +1.12910844, +1.12306840, +1.52672442],
        *[+1.40768172, +1.48154584, +1.31062963, +0.40374140, +0.75442607],
        *[+0.76482096, +0.98457281, +0.96702598, +1.05266584, +0.93274875],
        *[+1.04025281, +0.92738624, +1.07419210, +1.07900668, +1.04712861],
        *[+1.15018618, +1.15388455, +1.36313743, +1.36485106, +1.39801837],
        *[+1.18695346, +0.36273870, +0.58797255, +0.71961946, +0.96158233],
        *[+0.89585296, +0.81360499, +1.00794665, +0.92613682, +1.09152285],
        *[+1.14907070, +1.13508911, +1.08853785, +1.11005982, +1.12452195],
        *[+1.21642129, +1.36507125, +1.40340000, +1.16653482, +0.34125098],
        *[+0.58884173, +0.68441115, +0.56999999, +0.56999999, +0.56999999],
        *[+0.56999999, +0.56999999, +0.56999999, +0.56999999, +0.56999999],
        *[+0.56999999, +0.56999999, +0.56999999, +0.56999999, +0.56999999],
        *[+0.56999999, +0.87936784, +1.02761808, +0.93297476, +1.10172128],
        *[+0.97350071, +1.16695666, +1.23997927, +1.18464453, +1.14191734],
        *[+1.12334192, +1.01485321, +1.12950808, +1.30804834, +1.33689961],
        *[+1.27465977],
    ]
)

_eta2019 = torch.tensor(
    [
        *[+0.00000000],
        *[-0.35015861, +1.04121227, +0.09281243, +0.09412380, +0.26629137],
        *[+0.19408787, +0.05317918, +0.03151644, +0.32275132, +1.30996037],
        *[+0.24206510, +0.04147733, +0.11634126, +0.13155266, +0.15350650],
        *[+0.15250997, +0.17523529, +0.28774450, +0.42937314, +0.01896455],
        *[+0.07179178, -0.01121381, -0.03093370, +0.02716319, -0.01843812],
        *[-0.15270393, -0.09192645, -0.13418723, -0.09861139, +0.18338109],
        *[+0.08299615, +0.11370033, +0.19005278, +0.10980677, +0.12327841],
        *[+0.25345554, +0.58615231, +0.16093861, +0.04548530, -0.02478645],
        *[+0.01909943, +0.01402541, -0.03595279, +0.01137752, -0.03697213],
        *[+0.08009416, +0.02274892, +0.12801822, -0.02078702, +0.05284319],
        *[+0.07581190, +0.09663758, +0.09547417, +0.07803344, +0.64913257],
        *[+0.15348654, +0.05054344, +0.11000000, +0.11000000, +0.11000000],
        *[+0.11000000, +0.11000000, +0.11000000, +0.11000000, +0.11000000],
        *[+0.11000000, +0.11000000, +0.11000000, +0.11000000, +0.11000000],
        *[+0.11000000, -0.02786741, +0.01057858, -0.03892226, -0.04574364],
        *[-0.03874080, -0.03782372, -0.07046855, +0.09546597, +0.21953269],
        *[+0.02522348, +0.15263050, +0.08042611, +0.01878626, +0.08715453],
        *[+0.10500484],
    ]
)

_kcn2019 = torch.tensor(
    [
        *[+0.00000000],
        *[+0.04916110, +0.10937243, -0.12349591, -0.02665108, -0.02631658],
        *[+0.06005196, +0.09279548, +0.11689703, +0.15704746, +0.07987901],
        *[-0.10002962, -0.07712863, -0.02170561, -0.04964052, +0.14250599],
        *[+0.07126660, +0.13682750, +0.14877121, -0.10219289, -0.08979338],
        *[-0.08273597, -0.01754829, -0.02765460, -0.02558926, -0.08010286],
        *[-0.04163215, -0.09369631, -0.03774117, -0.05759708, +0.02431998],
        *[-0.01056270, -0.02692862, +0.07657769, +0.06561608, +0.08006749],
        *[+0.14139200, -0.05351029, -0.06701705, -0.07377246, -0.02927768],
        *[-0.03867291, -0.06929825, -0.04485293, -0.04800824, -0.01484022],
        *[+0.07917502, +0.06619243, +0.02434095, -0.01505548, -0.03030768],
        *[+0.01418235, +0.08953411, +0.08967527, +0.07277771, -0.02129476],
        *[-0.06188828, -0.06568203, -0.11000000, -0.11000000, -0.11000000],
        *[-0.11000000, -0.11000000, -0.11000000, -0.11000000, -0.11000000],
        *[-0.11000000, -0.11000000, -0.11000000, -0.11000000, -0.11000000],
        *[-0.11000000, -0.03585873, -0.03132400, -0.05902379, -0.02827592],
        *[-0.07606260, -0.02123839, +0.03814822, +0.02146834, +0.01580538],
        *[-0.00894298, -0.05864876, -0.01817842, +0.07721851, +0.07936083],
        *[+0.05849285],
    ]
)

_rad2019 = torch.tensor(
    [
        *[+0.00000000],
        *[+0.55159092, +0.66205886, +0.90529132, +1.51710827, +2.86070364],
        *[+1.88862966, +1.32250290, +1.23166285, +1.77503721, +1.11955204],
        *[+1.28263182, +1.22344336, +1.70936266, +1.54075036, +1.38200579],
        *[+2.18849322, +1.36779065, +1.27039703, +1.64466502, +1.58859404],
        *[+1.65357953, +1.50021521, +1.30104175, +1.46301827, +1.32928147],
        *[+1.02766713, +1.02291377, +0.94343886, +1.14881311, +1.47080755],
        *[+1.76901636, +1.98724061, +2.41244711, +2.26739524, +2.95378999],
        *[+1.20807752, +1.65941046, +1.62733880, +1.61344972, +1.63220728],
        *[+1.60899928, +1.43501286, +1.54559205, +1.32663678, +1.37644152],
        *[+1.36051851, +1.23395526, +1.65734544, +1.53895240, +1.97542736],
        *[+1.97636542, +2.05432381, +3.80138135, +1.43893803, +1.75505957],
        *[+1.59815118, +1.76401732, +1.63999999, +1.63999999, +1.63999999],
        *[+1.63999999, +1.63999999, +1.63999999, +1.63999999, +1.63999999],
        *[+1.63999999, +1.63999999, +1.63999999, +1.63999999, +1.63999999],
        *[+1.63999999, +1.47055223, +1.81127084, +1.40189963, +1.54015481],
        *[+1.33721475, +1.57165422, +1.04815857, +1.78342098, +2.79106396],
        *[+1.78160840, +2.47588882, +2.37670734, +1.76613217, +2.66172302],
        *[+2.82773085],
    ]
)


def solve(
    numbers: Tensor,
    positions: Tensor,
    total_charge: Tensor,
    model: ChargeModel,
    cn: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Solve the electronegativity equilibration for the partial charges minimizing
    the electrostatic energy.

    Parameters
    ----------
    numbers : Tensor
        Atomic numbers of all atoms in the system.
    positions : Tensor
        Cartesian coordinates of all atoms in the system.
    total_charge : Tensor
        Total charge of the system.
    model : ChargeModel
        Charge model to use.
    cn : Tensor
        Coordination numbers for all atoms in the system.

    Returns
    -------
    (Tensor, Tensor)
        Tuple of electrostatic energies and partial charges.

    Example
    -------
    >>> import torch
    >>> import xtbml.charges as charges
    >>> numbers = torch.tensor([7, 1, 1, 1])
    >>> positions=torch.tensor([
    ...     [+0.00000000000000, +0.00000000000000, -0.54524837997150],
    ...     [-0.88451840382282, +1.53203081565085, +0.18174945999050],
    ...     [-0.88451840382282, -1.53203081565085, +0.18174945999050],
    ...     [+1.76903680764564, +0.00000000000000, +0.18174945999050],
    ... ], requires_grad=True)
    >>> total_charge = torch.tensor(0.0, requires_grad=True)
    >>> cn = torch.tensor([3.0, 1.0, 1.0, 1.0])
    >>> eeq = charges.ChargeModel.param2019()
    >>> energy = torch.sum(charges.solve(numbers, positions, total_charge, eeq, cn)[0], -1)
    >>> energy.backward()
    >>> print(positions.grad)
    tensor([[-9.3132e-09,  7.4506e-09, -4.8064e-02],
            [-1.2595e-02,  2.1816e-02,  1.6021e-02],
            [-1.2595e-02, -2.1816e-02,  1.6021e-02],
            [ 2.5191e-02, -6.9849e-10,  1.6021e-02]])
    >>> print(total_charge.grad)
    tensor(0.6312)
    """

    if model.device != positions.device:
        raise RuntimeError(
            f"All tensors of '{model.__class__.__name__}' must be on the same "
            f"device!\nUse `{model.__class__.__name__}.param2019().to(device)` "
            "to correctly set the device."
        )

    if model.dtype != positions.dtype:
        raise RuntimeError(
            f"All tensors of '{model.__class__.__name__}' must have the same "
            f"dtype!\nUse `{model.__class__.__name__}.param2019().type(dtype)` "
            "to correctly set the dtype."
        )

    eps = positions.new_tensor(torch.finfo(positions.dtype).eps)

    real = real_atoms(numbers)
    mask = real_pairs(numbers, diagonal=True)

    distances = torch.where(
        mask,
        torch.cdist(positions, positions, p=2, compute_mode="use_mm_for_euclid_dist"),
        eps,
    )
    diagonal = mask.new_zeros(mask.shape)
    diagonal.diagonal(dim1=-2, dim2=-1).fill_(True)

    rhs = torch.concat(
        (
            -model.chi[numbers] + torch.sqrt(cn) * model.kcn[numbers],
            total_charge.unsqueeze(-1),
        ),
        dim=-1,
    )

    rad = model.rad[numbers]
    gamma = 1.0 / torch.sqrt(rad.unsqueeze(-1) ** 2 + rad.unsqueeze(-2) ** 2)
    eta = torch.where(
        real,
        model.eta[numbers] + torch.sqrt(torch.tensor(2.0 / math.pi)) / rad,
        distances.new_tensor(1.0),
    )
    coulomb = torch.where(
        diagonal,
        eta.unsqueeze(-1),
        torch.where(
            mask,
            torch.erf(distances * gamma) / distances,
            distances.new_tensor(0.0),
        ),
    )
    constraint = torch.where(
        real,
        distances.new_ones(numbers.shape),
        distances.new_zeros(numbers.shape),
    )
    zero = distances.new_zeros(numbers.shape[:-1])

    matrix = torch.concat(
        (
            torch.concat((coulomb, constraint.unsqueeze(-1)), dim=-1),
            torch.concat((constraint, zero.unsqueeze(-1)), dim=-1).unsqueeze(-2),
        ),
        dim=-2,
    )

    x = torch.linalg.solve(matrix, rhs)
    e = x * (0.5 * torch.einsum("...ij,...j->...i", matrix, x) - rhs)
    return e[..., :-1], x[..., :-1]
