"""
Molecules for testing the charges module.
"""

import torch

from dxtb.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    total_charge: Tensor
    """Reference values total charge of molecule."""

    q: Tensor
    """Atomic charges."""

    energy: Tensor
    """Atom-resolved electrostatic energy."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "NH3-dimer": {
        "total_charge": torch.tensor(0.0),
        "q": torch.tensor(
            [
                -0.8347351804,
                -0.8347351804,
                +0.2730523336,
                +0.2886305132,
                +0.2730523336,
                +0.2730523336,
                +0.2886305132,
                +0.2730523336,
            ],
        ),
        "energy": torch.tensor(
            [
                -0.5832575193,
                -0.5832575193,
                +0.1621643199,
                +0.1714161174,
                +0.1621643199,
                +0.1621643199,
                +0.1714161174,
                +0.1621643199,
            ],
        ),
    },
    "NH3": {
        "total_charge": torch.tensor(0.0),
        "q": torch.tensor([0.0]),
        "energy": torch.tensor([0.0]),
    },
    "PbH4-BiH3": {
        "total_charge": torch.tensor(0.0),
        "q": torch.tensor(
            [
                +0.1830965969,
                -0.0434600885,
                -0.0434600949,
                -0.0434600949,
                -0.0452680726,
                +0.0727632554,
                -0.0267371663,
                -0.0267371688,
                -0.0267371663,
            ]
        ),
        "energy": torch.tensor(
            [
                +0.1035379745,
                -0.0258195114,
                -0.0258195151,
                -0.0258195151,
                -0.0268938305,
                +0.0422307903,
                -0.0158831963,
                -0.0158831978,
                -0.0158831963,
            ],
        ),
    },
    "C6H5I-CH3SH": {
        "total_charge": torch.tensor(0.0),
        "q": torch.tensor(
            [
                -0.1029278713,
                -0.1001905841,
                -0.1028043772,
                -0.0774975738,
                -0.0007325498,
                -0.0780660341,
                -0.1962493355,
                +0.1120891066,
                +0.1205055899,
                +0.1123282728,
                +0.1197578368,
                +0.1121635250,
                -0.1711138357,
                +0.1212508178,
                -0.2031014175,
                +0.1153482095,
                +0.1143692362,
                +0.1048709842,
            ]
        ),
        "energy": torch.tensor(
            [
                -0.0666956672,
                -0.0649253132,
                -0.0666156432,
                -0.0501240988,
                -0.0004746778,
                -0.0504921903,
                -0.1274747615,
                +0.0665769222,
                +0.0715759533,
                +0.0667190716,
                +0.0711318128,
                +0.0666212167,
                -0.1116992442,
                +0.0720166288,
                -0.1300663998,
                +0.0685131245,
                +0.0679318540,
                +0.0622901437,
            ],
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
