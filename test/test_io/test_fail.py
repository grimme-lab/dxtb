"""
Test input files that are not in correct format.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dxtb import io


def test_xyz() -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", "atom.xyz")
    with pytest.raises(ValueError):
        io.read_structure_from_file(p)


@pytest.mark.parametrize("file", ["mol.json", "mol2.json", "mol3.json"])
def test_json(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", file)
    with pytest.raises(KeyError):
        io.read_structure_from_file(p)


@pytest.mark.parametrize("file", ["coord"])
def test_coord(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", file)
    with pytest.raises(ValueError):
        io.read_structure_from_file(p)


@pytest.mark.parametrize("file", ["energy", "energy2", "energy3"])
def test_read_tm_energy(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", file)
    with pytest.raises(ValueError):
        io.read_tm_energy(p)


@pytest.mark.parametrize("file", ["orca.engrad", "orca2.engrad"])
def test_read_orca_engrad(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", file)
    with pytest.raises(ValueError):
        io.read_orca_engrad(p)


def test_file_not_found() -> None:
    p = Path(Path(__file__).parent.resolve(), "wrong", "notfound")
    with pytest.raises(FileNotFoundError):
        io.read_orca_engrad(p)

    with pytest.raises(FileNotFoundError):
        io.read_tm_energy(p)

    with pytest.raises(FileNotFoundError):
        io.read_tblite_gfn(p)

    with pytest.raises(FileNotFoundError):
        io.read_coord(p)

    with pytest.raises(FileNotFoundError):
        io.read_qcschema(p)

    with pytest.raises(FileNotFoundError):
        io.read_xyz(p)

    with pytest.raises(FileNotFoundError):
        io.read_structure_from_file(p)
