"""
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""

import torch

from dxtb.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    escf: Tensor
    """SCF energy for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "escf": torch.tensor(-4.0142947446183e-01),
    },
    "C": {
        "escf": torch.tensor(-1.7411359557542),
    },
    "Rn": {
        "escf": torch.tensor(-3.6081562853046),
    },
    "H2": {
        "escf": torch.tensor(-1.0585984032484),
    },
    "LiH": {
        "escf": torch.tensor(-0.88306406116865),
    },
    "HLi": {
        "escf": torch.tensor(0.0),
    },
    "HC": {
        "escf": torch.tensor(0.0),
    },
    "HHe": {
        "escf": torch.tensor(0.0),
    },
    "S2": {
        "escf": torch.tensor(-7.3285116888517),
    },
    "H2O": {
        "escf": torch.tensor(-5.8052489623704e00),
    },
    "CH4": {
        "escf": torch.tensor(-4.3393059719255e00),
    },
    "SiH4": {
        "escf": torch.tensor(-4.0384093532453),
    },
    "PbH4-BiH3": {
        "escf": torch.tensor(-7.6074262079844),
    },
    "C6H5I-CH3SH": {
        "escf": torch.tensor(-27.612142805843),
    },
    "MB16_43_01": {
        "escf": torch.tensor(-33.200116717478),
    },
    "LYS_xao": {
        "escf": torch.tensor(-48.850798066902),
    },
    "C60": {
        "escf": torch.tensor(-128.79148324775),
    },
    "vancoh2": {
        "escf": torch.tensor(-3.2618651888175e02),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
