"""
Test reading files from other programs.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from dxtb import io


@pytest.mark.parametrize("file", ["tblite.json"])
def test_read_tblite_json(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "outer", file)
    data = io.read_tblite_gfn(p)

    assert data["version"] == "0.2.1"
    assert data["energy"] == -0.8814248363751351
    assert data["energies"] == [-0.30036182118846183, -0.5810630151866734]
    assert data["gradient"] == [
        4.8212532360532058e-18,
        -8.5642661603437023e-33,
        -1.2967437449579400e-02,
        -4.8212532360532058e-18,
        8.5642661603437023e-33,
        1.2967437449579402e-02,
    ]
    assert data["virial"] == [
        0.0000000000000000e00,
        0.0000000000000000e00,
        -7.2702928950083074e-18,
        0.0000000000000000e00,
        0.0000000000000000e00,
        1.2914634508491054e-32,
        -7.2702928950083074e-18,
        1.2914634508491054e-32,
        3.9108946881752781e-02,
    ]


@pytest.mark.parametrize("file", ["orca.engrad"])
def test_read_orca_engrad(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "outer", file)
    e, g = io.read_orca_engrad(p)

    assert e == -17.271065945172
    assert g == [
        [-3.2920000059277754e-09, 0.0, -0.006989939603954554],
        [-0.0014856194611638784, -0.0, +0.0035081561654806137],
        [+0.0014856194611638784, -0.0, +0.0035081561654806137],
    ]


@pytest.mark.parametrize("file", ["energy"])
def test_read_tm_energy(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "outer", file)
    e = io.read_tm_energy(p)

    assert e == -291.856093690170


@pytest.mark.parametrize("file", ["orca.engrad"])
def test_fail_read_tm_energy(file: str) -> None:
    p = Path(Path(__file__).parent.resolve(), "outer", file)
    with pytest.raises(ValueError):
        io.read_tm_energy(p)
