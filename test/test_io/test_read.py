"""
Test reading files.
"""

from pathlib import Path

import pytest
import torch

from dxtb import io


@pytest.mark.parametrize("file", ["mol.xyz", "mol.json", "coord"])
def test_types(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "files", file)

    numbers, positions = io.read_structure_from_file(p)

    assert numbers == [8, 1, 1]

    ref_positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ]
    )
    positions = torch.tensor(positions)
    assert torch.allclose(positions, ref_positions)


@pytest.mark.parametrize("file", ["mol.mol", "mol.ein", "POSCAR"])
def test_fail(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "files", file)

    with pytest.raises(NotImplementedError):
        io.read_structure_from_file(p)
