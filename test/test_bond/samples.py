"""
Molecules for testing the charges module.
"""
from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    bo: Tensor
    """Reference bond orders."""

    cn: Tensor
    """DFT-D3 coordination number."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "PbH4-BiH3": {
        "bo": torch.tensor(
            [
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.6533,
                0.6533,
                0.6533,
                0.6499,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
                0.5760,
            ],
        ),
        "cn": torch.tensor(
            [
                3.9388208389,
                0.9832025766,
                0.9832026958,
                0.9832026958,
                0.9865897894,
                2.9714603424,
                0.9870455265,
                0.9870456457,
                0.9870455265,
            ],
        ),
    },
    "C6H5I-CH3SH": {
        "bo": torch.tensor(
            [
                0.4884,
                0.5180,
                0.4006,
                0.4884,
                0.4884,
                0.4012,
                0.4884,
                0.5181,
                0.4006,
                0.5181,
                0.5144,
                0.4453,
                0.5144,
                0.5145,
                0.4531,
                0.5180,
                0.5145,
                0.4453,
                0.4531,
                0.4012,
                0.4453,
                0.4006,
                0.4453,
                0.4006,
                0.6041,
                0.3355,
                0.6041,
                0.3355,
                0.5645,
                0.5673,
                0.5670,
                0.5645,
                0.5673,
                0.5670,
            ]
        ),
        "cn": torch.tensor(
            [
                3.1393690109,
                3.1313166618,
                3.1393768787,
                3.3153429031,
                3.1376547813,
                3.3148119450,
                1.5363609791,
                1.0035246611,
                1.0122337341,
                1.0036621094,
                1.0121959448,
                1.0036619902,
                2.1570565701,
                0.9981809855,
                3.9841127396,
                1.0146225691,
                1.0123561621,
                1.0085891485,
            ],
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
