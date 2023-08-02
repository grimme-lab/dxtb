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
    """Dipole moment of molecule."""

    freq: Tensor
    """Frequencies for GFN1-xTB."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "dipole": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "H2": {
        "dipole": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "LiH": {
        "dipole": torch.tensor([0.0000, 0.0000, -2.4794], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "HHe": {
        "dipole": torch.tensor([0.0000, 0.0000, 0.2565], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "H2O": {
        "dipole": torch.tensor([-0.0000, -0.0000, 1.1208], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "CH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "SiH4": {
        "dipole": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "PbH4-BiH3": {
        "dipole": torch.tensor([-0.0000, -1.0555, 0.0000], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "MB16_43_01": {
        "dipole": torch.tensor([0.2903, -1.0541, -2.0211], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "LYS_xao": {
        "dipole": torch.tensor([-1.0011, -1.6512, -0.7423], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "C60": {
        "dipole": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
    "vancoh2": {
        "dipole": torch.tensor([2.4516, 8.1274, 0.3701], dtype=torch.float64),
        "freq": torch.tensor([], dtype=torch.float64),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
