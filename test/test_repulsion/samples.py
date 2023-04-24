"""
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""
from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    gfn1: Tensor
    """Reference values for GFN1-xTB"""

    gfn1_grad: Tensor
    """Reference values for GFN1-xTB gradients"""

    gfn2: Tensor
    """Reference values for GFN1-xTB"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H2": {
        "gfn1": torch.tensor(2.2362490584964e-02),
        "gfn2": torch.tensor(3.8770781236977e-02),
        "gfn1_grad": torch.tensor([]),
    },
    "H2O": {
        "gfn1": torch.tensor(3.6764721202060e-02),
        "gfn2": torch.tensor(3.3793519342311e-02),
        "gfn1_grad": torch.tensor(
            [
                [
                    +0.0000000000000000,
                    +0.0000000000000000,
                    +0.10833611039049376,
                ],
                [
                    +6.9798837064846950e-002,
                    +0.0000000000000000,
                    -5.4168055195246881e-002,
                ],
                [
                    -6.9798837064846950e-002,
                    +0.0000000000000000,
                    -5.4168055195246881e-002,
                ],
            ]
        ),
    },
    "SiH4": {
        "gfn1": torch.tensor(3.0331305861808766e-002),
        "gfn2": torch.tensor(3.1536555053538279e-002),
        "gfn1_grad": torch.tensor(
            [
                [
                    +0.0000000000000000,
                    +0.0000000000000000,
                    +0.0000000000000000,
                ],
                [
                    -1.7473488309747563e-002,
                    -1.7473488309747563e-002,
                    +1.7473488309747563e-002,
                ],
                [
                    +1.7473488309747563e-002,
                    +1.7473488309747563e-002,
                    +1.7473488309747563e-002,
                ],
                [
                    -1.7473488309747563e-002,
                    +1.7473488309747563e-002,
                    -1.7473488309747563e-002,
                ],
                [
                    +1.7473488309747563e-002,
                    -1.7473488309747563e-002,
                    -1.7473488309747563e-002,
                ],
            ]
        ),
    },
    "ZnOOH-": {
        "gfn1": torch.tensor(2.9095479886131e-02),
        "gfn1_grad": torch.tensor(0.0),
        "gfn2": torch.tensor(2.2289239363144e-02),
    },
    "LYS_xao": {
        "gfn1": torch.tensor(0.54175667737478617),
        "gfn1_grad": torch.tensor(0.0),
        "gfn2": torch.tensor(5.5376567296060e-01),
    },
    "MB16_43_01": {
        "gfn1": torch.tensor(0.16777923624986593),
        "gfn1_grad": torch.tensor(0.0),
        "gfn2": torch.tensor(0.15297938789402879),
    },
    "MB16_43_02": {
        "gfn1": torch.tensor(0.12702003611285190),
        "gfn1_grad": torch.tensor(0.0),
        "gfn2": torch.tensor(0.10745931926703985),
    },
    "MB16_43_03": {
        "gfn1": torch.tensor(0.16600531760459214),
        "gfn1_grad": torch.tensor(0.0),
        "gfn2": torch.tensor(0.15818907118271672),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
