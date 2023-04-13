"""
Test reading files.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dxtb import io
from dxtb.constants import AA2AU


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


def test_xyz_coord() -> None:
    """Check if unit conversion is correct (coord: a.u., xyz: Angström)."""
    p1 = Path(Path(__file__).parent.resolve(), "files", "coord")
    p2 = Path(Path(__file__).parent.resolve(), "files", "mol.xyz")

    nums1, pos1 = io.read_structure_from_file(p1, ftype="turbomole")
    nums2, pos2 = io.read_structure_from_file(p2)

    pos1 = torch.tensor(pos1)
    pos2 = torch.tensor(pos2)

    assert nums1 == [8, 1, 1]
    assert nums2 == [8, 1, 1]

    ref_positions = torch.tensor(
        [
            [+0.00000000000000, +0.00000000000000, -0.74288549752983],
            [-1.43472674945442, +0.00000000000000, +0.37144274876492],
            [+1.43472674945442, +0.00000000000000, +0.37144274876492],
        ]
    )

    assert torch.allclose(pos1, pos2)
    assert torch.allclose(pos1, ref_positions)
    assert torch.allclose(pos2, ref_positions)


@pytest.mark.parametrize("file", ["mol.mol", "mol.ein", "POSCAR"])
def test_fail(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "files", file)

    with pytest.raises(NotImplementedError):
        io.read_structure_from_file(p)


def test_read_empty() -> None:
    p = Path(Path(__file__).parent.resolve(), "files", "empty.xyz")

    with pytest.raises(ValueError):
        io.read_structure_from_file(p)


def test_read_atom() -> None:
    """
    Read a single atom placed at [0.0, 0.0, 0.0]. The reader should modify the
    x-coordinate to avoid a clash with zero-padding.
    """
    p = Path(Path(__file__).parent.resolve(), "files", "atom.xyz")
    nums, pos = io.read_structure_from_file(p)

    assert nums == [1]
    assert (torch.tensor(pos) == torch.tensor([1.0, 0.0, 0.0])).all()


def test_read_atom2() -> None:
    """
    Read a single atom placed at [0.3, 0.3, 0.3].
    """
    p = Path(Path(__file__).parent.resolve(), "files", "atom2.xyz")
    nums, pos = io.read_structure_from_file(p)
    ref = torch.tensor([[0.3, 0.3, 0.3]]) * AA2AU

    assert nums == [2]
    assert pytest.approx(ref) == torch.tensor(pos)


def test_fail_last_zero() -> None:
    """
    Read a molecule where the last atom placed at [0.0, 0.0, 0.0]. This would
    clash with zero-padding; hence, we immediately throw an error.
    """
    p = Path(Path(__file__).parent.resolve(), "files", "lastzero.xyz")
    with pytest.raises(ValueError):
        io.read_structure_from_file(p)


def test_fail_file_not_found() -> None:
    p = Path(Path(__file__).parent.resolve(), "files", "filenotfound.xyz")
    with pytest.raises(FileNotFoundError):
        io.read_structure_from_file(p)


def test_fail_unknown_ftype() -> None:
    p = Path(Path(__file__).parent.resolve(), "files", "mol.xyz")
    with pytest.raises(ValueError):
        io.read_structure_from_file(p, ftype="unknown")
