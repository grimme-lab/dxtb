"""
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""

import torch

from ..molecules import merge_nested_dicts, mols

from xtbml.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    charge: Tensor
    """Total charge of the molecule"""

    escf: Tensor
    """SCF energy for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "Ag2Cl22-": {
        "charge": torch.tensor(-2.0),
        "escf": torch.tensor(-2.5297870091005e01),
    },
    "Al3+Ar6": {
        "charge": torch.tensor(3.0),
        "escf": torch.tensor(-3.6303223981129e01),
    },
    "AD7en+": {
        "charge": torch.tensor(1.0),
        "escf": torch.tensor(-4.3226840214360e01),
    },
    "C2H4F+": {
        "charge": torch.tensor(1.0),
        "escf": torch.tensor(-1.1004178291636e01),
    },
    "ZnOOH-": {
        "charge": torch.tensor(-1.0),
        "escf": torch.tensor(-1.0913986485487e01),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
