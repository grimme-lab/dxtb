"""
Data for testing Coulomb contribution.
"""

from __future__ import annotations

import torch

from dxtb._types import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    energy: Tensor
    """Single-point energy with electric field of (-2, 0, 0)."""

    energy_monopole: Tensor
    """
    Single-point energy with electric field but neglecting dipole contributions.
    """

    energy_no_field: Tensor
    """Single-point energy without electric field."""


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "LiH": {
        "energy": torch.tensor(
            -0.89252528137221931,
            dtype=torch.float64,
        ),
        "energy_monopole": torch.tensor(
            -0.88142483639196112,
            dtype=torch.float64,
        ),
        "energy_no_field": torch.tensor(
            -0.88142483639196123,
            dtype=torch.float64,
        ),
    },
    "SiH4": {
        "energy": torch.tensor(
            -4.0315642270838108,
            dtype=torch.float64,
        ),
        "energy_monopole": torch.tensor(
            -4.019892923534051,
            dtype=torch.float64,
        ),
        "energy_no_field": torch.tensor(
            -4.0087585461132633,
            dtype=torch.float64,
        ),
    },
    "MB16_43_01": {
        "energy": torch.tensor(
            -33.092759826817499,
            dtype=torch.float64,
        ),
        "energy_monopole": torch.tensor(
            -33.08278119794784,
            dtype=torch.float64,
        ),
        "energy_no_field": torch.tensor(
            -33.040345115781605,
            dtype=torch.float64,
        ),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
