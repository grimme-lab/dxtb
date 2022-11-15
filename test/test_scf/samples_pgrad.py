"""
Test data for gradient of parameters w.r.t. forces. The gradient is calculated
within dxtb using full gradient tracking in the SCF and `torch.float`.
"""

import torch

from dxtb.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    kcn: Tensor
    """Gradient for `kcn` parameter w.r.t. forces."""

    selfenergy: Tensor
    """Gradient for `selfenergy` parameter w.r.t. forces."""

    shpoly: Tensor
    """Gradient for `shpoly` parameter w.r.t. forces."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LiH": {
        "selfenergy": torch.tensor(
            [+0.0002029369, +0.0017547115, +0.1379896402, -0.1265652627]
        ),
        "kcn": torch.tensor(
            [-0.1432282478, -0.0013212233, -0.1811404824, +0.0755317509]
        ),
        "shpoly": torch.tensor(
            [+0.0408593193, -0.0007219329, -0.0385218151, +0.0689999014]
        ),
    },
    "H2O": {
        "selfenergy": torch.tensor(
            [
                +0.0201908200979233,
                -0.1104588359594345,
                -0.1022646874189377,
                +0.3026777505874634,
            ]
        ),
        "kcn": torch.tensor(
            [
                -0.0972348898649216,
                +0.1041719615459442,
                +0.0487641692161560,
                -1.0812859535217285,
            ]
        ),
        "shpoly": torch.tensor(
            [
                -0.0837635546922684,
                +0.0743020549416542,
                -0.3784617185592651,
                +0.2503065466880798,
            ]
        ),
    },
    "SiH4": {
        "selfenergy": torch.tensor(
            [
                +5.2154064178466797e-08,
                +4.9360096454620361e-08,
                +9.6857547760009766e-08,
                +6.3329935073852539e-08,
                -5.5879354476928711e-08,
            ]
        ),
        "kcn": torch.tensor(
            [
                -2.3841857910156250e-07,
                -5.4948031902313232e-08,
                -3.5340434578756685e-07,
                -2.0341597917195031e-07,
                +2.2979276081969147e-07,
            ]
        ),
        "shpoly": torch.tensor(
            [
                +1.6018748283386230e-07,
                -2.3283064365386963e-08,
                +2.2165477275848389e-07,
                -1.1362135410308838e-07,
                +1.2805685400962830e-08,
            ]
        ),
    },
    "LYS_xao": {
        "selfenergy": torch.tensor(
            [
                -0.0236866232007742,
                +0.0329584367573261,
                -0.0796709954738617,
                +0.0810616388916969,
                -0.0167455300688744,
                -0.0992878675460815,
                +0.0035702660679817,
                -0.0344504565000534,
            ]
        ),
        "kcn": torch.tensor(
            [
                +0.0371370315551758,
                -0.0326246172189713,
                +0.2156552672386169,
                -1.0895895957946777,
                -0.2504942417144775,
                -0.5578549504280090,
                -0.0560819841921329,
                -0.1159448027610779,
            ]
        ),
        "shpoly": torch.tensor(
            [
                +0.2418332993984222,
                -0.0145443072542548,
                +0.0408689938485622,
                -0.1285742074251175,
                +0.0428046323359013,
                -0.0263019893318415,
                -0.0102892229333520,
                +0.0534406043589115,
            ]
        ),
    },
}


refs_pgrad: dict[str, Record] = merge_nested_dicts(mols, refs)
