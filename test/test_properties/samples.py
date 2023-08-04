"""
Reference data for property calculations.
"""
from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records."""

    dipole: Tensor
    """Dipole moment of molecule with (0, 0, 0) field."""

    dipole2: Tensor
    """Dipole moment of molecule with (-2, 1, 0.5) field."""

    quad: Tensor
    """Quadrupole moment of molecule with (0, 0, 0) field."""

    quad2: Tensor
    """Quadrupole moment of molecule with (2, 3, 5) field."""

    freq: Tensor
    """Frequencies for GFN1-xTB."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([0.0, 0.0, 0.0]),
        "quad": torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
        "quad2": torch.tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]),
        "freq": torch.tensor([]),
    },
    "H2": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([0.0000, 0.0000, 0.1015]),
        "quad": torch.tensor([-0.1591, 0.0000, -0.1591, 0.0000, 0.0000, 0.3183]),
        "quad2": torch.tensor([-0.1584, 0.0000, -0.1584, 0.0000, 0.0000, 0.3168]),
        "freq": torch.tensor([]),
    },
    "LiH": {
        "dipole": torch.tensor([0.0000, 0.0000, -2.4794]),
        "dipole2": torch.tensor([-1.2293, 0.3073, -1.1911]),
        "quad": torch.tensor([-0.6422, 0.0000, -0.6422, -0.0000, 0.0000, 1.2843]),
        "quad2": torch.tensor([-2.0799, 0.4581, -0.3620, 3.7017, -0.9254, 2.4419]),
        "freq": torch.tensor([]),
    },
    "HHe": {
        "dipole": torch.tensor([0.0000, 0.0000, 0.2565]),
        "dipole2": torch.tensor([0.0000, 0.0000, 0.2759]),
        "quad": torch.tensor([-0.2259, 0.0000, -0.2259, 0.0000, 0.0000, 0.4517]),
        "quad2": torch.tensor([-0.2465, 0.0000, -0.2465, 0.0000, 0.0000, 0.4929]),
        "freq": torch.tensor([]),
    },
    "H2O": {
        "dipole": torch.tensor([-0.0000, -0.0000, 1.1208]),
        "dipole2": torch.tensor([-0.1418, 0.0006, 1.1680]),
        "quad": torch.tensor([2.2898, 0.0000, -0.8549, 0.0000, 0.0000, -1.4349]),
        "quad2": torch.tensor([2.3831, 0.0001, -0.8838, -0.2055, -0.0022, -1.4993]),
        "freq": torch.tensor([]),
    },
    "CH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-0.2230, 0.0480, 0.1661]),
        "quad": torch.tensor([0.0000, 0.0000, -0.0000, 0.0000, -0.0000, 0.0000]),
        "quad2": torch.tensor([-0.0003, -0.6487, 0.0003, -0.1852, 0.8712, -0.0000]),
        "freq": torch.tensor([]),
    },
    "SiH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-1.2136, 0.2518, 0.8971]),
        "quad": torch.tensor([-0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 0.0000]),
        "quad2": torch.tensor([0.1276, -0.5236, -0.1362, -0.1603, 0.6765, 0.0085]),
        "freq": torch.tensor([]),
    },
    "PbH4-BiH3": {
        "dipole": torch.tensor([-0.0000, -1.0555, 0.0000]),
        "dipole2": torch.tensor([-1.7609, -0.6177, 1.3402]),
        "quad": torch.tensor([5.3145, -0.0000, -10.6290, -0.0000, -0.0000, 5.3145]),
        "quad2": torch.tensor([4.8044, -0.9509, -10.4691, -0.7061, 0.2940, 5.6647]),
        "freq": torch.tensor([]),
    },
    "MB16_43_01": {
        "dipole": torch.tensor([0.2903, -1.0541, -2.0211]),
        "dipole2": torch.tensor([-5.3891, 1.2219, 2.6970]),
        "quad": torch.tensor([6.0210, 26.8833, -4.7743, 24.5314, -35.8644, -1.2467]),
        "quad2": torch.tensor(
            [7.8772, 22.4950, -0.3045, 26.1071, -34.4454, -7.5727],
            dtype=torch.float64,
        ),
        "freq": torch.tensor([]),
    },
    "LYS_xao": {
        "dipole": torch.tensor([-1.0012, -1.6513, -0.7423]),
        "dipole2": torch.tensor([-14.0946, 0.3401, 2.1676]),
        "quad": torch.tensor([-7.9018, -15.7437, 11.1642, 8.3550, 21.6634, -3.2624]),
        "quad2": torch.tensor(
            [-68.4795, 7.1007, 49.4005, 19.7302, 23.4305, 19.0790],
        ),
        "freq": torch.tensor([]),
    },
    "C60": {
        "dipole": torch.tensor([0.0, 0.0, 0.0]),
        "dipole2": torch.tensor([-20.5633, 5.3990, 15.6925]),
        "quad": torch.tensor([0.0000, 0.0000, -0.0000, 0.0000, 0.0000, -0.0000]),
        "quad2": torch.tensor([-0.3883, -5.9370, -1.4709, -6.8867, 2.8313, 1.8591]),
        "freq": torch.tensor([]),
    },
    # "vancoh2": {
    # "dipole": torch.tensor([2.4516, 8.1274, 0.3701]),
    # "dipole2": torch.tensor([-81.9131 ,   22.9779   , 42.5335]),
    # "quad": torch.tensor([-24.1718,   -11.7343 ,   37.5885 ,  -28.6426  ,  20.7013   ,-13.4167]),
    # "quad2": torch.tensor([-56.4013, -713.0469 ,  80.4179   , 58.0042  , 164.6469  , -24.0167]),
    # "freq": torch.tensor([]),
    # },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
