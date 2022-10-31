"""
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""

import torch

from xtbml.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    gfn1: Tensor
    """Reference values for GFN1-xTB"""

    gfn2: Tensor
    """Reference values for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LYS_xao": {
        "gfn1": torch.tensor(0.54175667737478617),
        "gfn2": torch.tensor(0.0),
    },
    "MB16_43_01": {
        "gfn1": torch.tensor(0.16777923624986593),
        "gfn2": torch.tensor(0.15297938789402879),
    },
    "MB16_43_02": {
        "gfn1": torch.tensor(0.12702003611285190),
        "gfn2": torch.tensor(0.10745931926703985),
    },
    "MB16_43_03": {
        "gfn1": torch.tensor(0.16600531760459214),
        "gfn2": torch.tensor(0.15818907118271672),
    },
    "SiH4": {
        "gfn1": torch.tensor(3.0331305861808766e-002),
        "gfn2": torch.tensor(3.1536555053538279e-002),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
