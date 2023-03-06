"""
Reference single point energies (from tblite).
"""
from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """
    Format of reference records containing GFN1-xTB and GFN2-xTB reference values.
    """

    etot: Tensor
    """Total energy for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "etot": torch.tensor(-4.0142947446183e-01),
    },
    "H2": {
        "etot": torch.tensor(-1.0362714373390e00),
    },
    "H2O": {
        "etot": torch.tensor(-5.7686218257620e00),
    },
    "NO2": {
        "etot": torch.tensor(-1.2409798675060e01),
    },
    "CH4": {
        "etot": torch.tensor(-4.2741992424931e00),
    },
    "SiH4": {
        "etot": torch.tensor(-4.0087585461086e00),
    },
    "LYS_xao": {
        "etot": torch.tensor(-4.8324739766346e01),
    },
    "C60": {
        "etot": torch.tensor(-1.2673081838911e02),
    },
    "vancoh2": {
        "etot": torch.tensor(-3.2295379428673e02),
    },
    "AD7en+": {
        "etot": torch.tensor(-4.2547841532513e01),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
