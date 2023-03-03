"""
Samples for test of halogen bond correction.
"""
from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records."""

    energy: Tensor
    """Reference value for energy from halogen bond correction."""

    gradient: Tensor
    """Reference nuclear gradient."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "br2nh3": {
        "energy": torch.tensor(2.4763110097465683e-3),
        "gradient": torch.tensor(
            [
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    -1.3105758671001298e-02,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +1.3105758671001298e-02,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
            ]
        ),
    },
    "br2nh2o": {
        "energy": torch.tensor(1.0010592532310653e-003),
        "gradient": torch.tensor([]),
    },
    "br2och2": {
        "energy": torch.tensor(-6.7587305781592112e-4),
        "gradient": torch.tensor([]),
    },
    "finch": {
        "energy": torch.tensor(1.1857937381795408e-2),
        "gradient": torch.tensor(
            [
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    -2.0420986552797993e-02,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +2.0420986552797993e-02,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
                [
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                    +0.0000000000000000e00,
                ],
            ]
        ),
    },
    "tmpda": {
        "energy": torch.tensor(7.6976121430560651e-002),
        "gradient": torch.tensor([]),
    },
    "tmpda_mod": {
        "energy": torch.tensor(3.1574395196210699e-003),
        "gradient": torch.tensor([]),
    },
    "LYS_xao": {
        "energy": torch.tensor(0.0),
        "gradient": torch.tensor([]),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
