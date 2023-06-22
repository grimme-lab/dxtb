"""
Test the packing utility functions.
"""

import torch

from dxtb.utils.batch import pack

mol1 = torch.tensor([1, 1])  # H2
mol2 = torch.tensor([8, 1, 1])  # H2O
mol3 = torch.tensor([6, 1, 1, 1, 1])  # CH4


def test_single_tensor() -> None:
    # dummy test: only give single tensor
    assert (mol1 == pack(mol1)).all()


def test_standard() -> None:
    # standard packing
    ref = torch.tensor(
        [
            [1, 1, 0],  # H2
            [8, 1, 1],  # H2O
        ],
    )
    packed = pack([mol1, mol2])
    assert (packed == ref).all()


def test_axis() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0, 0, 0],  # H2
            [8, 1, 1, 0, 0],  # H2O
            [6, 1, 1, 1, 1],  # CH4
        ],
    )
    ref_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    # different axis
    packed, mask = pack([mol1, mol2, mol3], axis=-1, return_mask=True)
    assert (packed == ref.T).all()
    assert (mask == ref_mask.T).all()


def test_size() -> None:
    ref = torch.tensor(
        [
            [1, 1, 0, 0],  # H2
            [8, 1, 1, 0],  # H2O
        ],
    )

    # one additional column of padding
    packed = pack([mol1, mol2], size=[4])
    assert (packed == ref).all()
