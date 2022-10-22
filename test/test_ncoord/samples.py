"""
Data for testing repulsion taken from https://github.com/grimme-lab/mstore.
"""

import torch

from ..molecules import merge_nested_dicts, mols

from xtbml.typing import Molecule, Tensor, TypedDict


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    cn: Tensor
    """DFT-D3 coordination number"""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "PbH4-BiH3": {
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
            ]
        )
    },
    "C6H5I-CH3SH": {
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
            ]
        )
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
