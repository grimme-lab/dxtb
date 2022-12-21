"""
Data for SCF energies.
"""

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    escf: Tensor
    """SCF energy for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "escf": torch.tensor(-4.0142947446183e-01, dtype=torch.float64),
    },
    "C": {
        "escf": torch.tensor(-1.7411359557542, dtype=torch.float64),
    },
    "Rn": {
        "escf": torch.tensor(-3.6081562853046, dtype=torch.float64),
    },
    "H2": {
        "escf": torch.tensor(-1.0585984032484, dtype=torch.float64),
    },
    "LiH": {
        "escf": torch.tensor(-0.88306406116865, dtype=torch.float64),
    },
    "HLi": {
        "escf": torch.tensor(-0.88306406116865, dtype=torch.float64),
    },
    "HC": {
        "escf": torch.tensor(0.0, dtype=torch.float64),
    },
    "HHe": {
        "escf": torch.tensor(0.0, dtype=torch.float64),
    },
    "S2": {
        "escf": torch.tensor(-7.3285116888517, dtype=torch.float64),
    },
    "H2O": {
        "escf": torch.tensor(-5.8052489623704e00, dtype=torch.float64),
    },
    "CH4": {
        "escf": torch.tensor(-4.3393059719255e00, dtype=torch.float64),
    },
    "SiH4": {
        "escf": torch.tensor(-4.0384093532453, dtype=torch.float64),
    },
    "PbH4-BiH3": {
        "escf": torch.tensor(-7.6074262079844, dtype=torch.float64),
    },
    "C6H5I-CH3SH": {
        "escf": torch.tensor(-27.612142805843, dtype=torch.float64),
    },
    "MB16_43_01": {
        "escf": torch.tensor(-33.200116717478, dtype=torch.float64),
    },
    "LYS_xao": {
        "escf": torch.tensor(-48.850798066902, dtype=torch.float64),
    },
    "LYS_xao_dist": {
        "escf": torch.tensor(-47.020544162958, dtype=torch.float64),
    },
    "C60": {
        "escf": torch.tensor(-128.79148324775, dtype=torch.float64),
    },
    "vancoh2": {
        "escf": torch.tensor(-3.2618651888175e02, dtype=torch.float64),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
