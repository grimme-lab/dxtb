# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test the parametrization of the Hamiltonian.
"""
# pylint: disable=missing-function-docstring, protected-access
from __future__ import annotations

import tempfile as td
from pathlib import Path

import pytest
import tomli as toml
import torch
from tad_mctc.convert import symbol_to_number

from dxtb import GFN1_XTB, GFN2_XTB, Calculator
from dxtb._src.param import Param
from dxtb._src.param.meta import Meta
from dxtb._src.typing import DD

try:
    import tomli_w as toml_w  # type: ignore
except ImportError:
    try:
        import toml as toml_w  # type: ignore
    except ImportError:
        toml_w = None

try:
    import yaml
except ImportError:
    yaml = None

from ..conftest import DEVICE


def test_builtin_gfn1() -> None:
    par = GFN1_XTB.model_copy(deep=True)

    assert isinstance(par.meta, Meta)
    assert par.meta.name == "GFN1-xTB"
    assert par.meta.version == 1

    assert par.dispersion is not None
    assert par.repulsion is not None
    assert par.charge is not None
    assert par.multipole is None
    assert par.halogen is not None
    assert par.thirdorder is not None
    assert par.hamiltonian is not None

    assert par.hamiltonian.xtb.cn == "exp"
    assert "sp" in par.hamiltonian.xtb.shell
    assert par.hamiltonian.xtb.wexp == 0.0
    assert par.hamiltonian.xtb.kpol == 2.85
    assert par.hamiltonian.xtb.enscale == -7.0e-3

    assert "Te" in par.element


def test_param_minimal() -> None:
    data = """
    [hamiltonian.xtb]
    wexp = 5.0000000000000000E-01
    kpol = 2.0000000000000000E+00
    enscale = 2.0000000000000000E-02
    cn = "gfn"
    shell = {ss=1.85, pp=2.23, dd=2.23, sd=2.00, pd=2}
    [element.H]
    shells = [ "1s" ]
    levels = [ -1.0707210999999999E+01 ]
    slater = [ 1.2300000000000000E+00 ]
    ngauss = [ 3 ]
    refocc = [ 1.0000000000000000E+00 ]
    shpoly = [ -9.5361800000000000E-03 ]
    kcn = [ -5.0000000000000003E-02 ]
    gam = 4.0577099999999999E-01
    lgam = [ 1.0000000000000000E+00 ]
    gam3 = 8.0000000000000016E-02
    zeff = 1.1053880000000000E+00
    arep = 2.2137169999999999E+00
    xbond = 0.0000000000000000E+00
    dkernel = 5.5638889999999996E-02
    qkernel = 2.7430999999999999E-04
    mprad = 1.3999999999999999E+00
    mpvcn = 1.0000000000000000E+00
    en = 2.2000000000000002E+00
    [element.C]
    shells = [ "2s", "2p" ]
    levels = [ -1.3970922000000002E+01, -1.0063292000000001E+01 ]
    slater = [ 2.0964320000000001E+00, 1.8000000000000000E+00 ]
    ngauss = [ 4, 4 ]
    refocc = [ 1.0000000000000000E+00, 3.0000000000000000E+00 ]
    shpoly = [ -2.2943210000000002E-02, -2.7110200000000002E-03 ]
    kcn = [ -1.0214400000000000E-02, 1.6165700000000002E-02 ]
    gam = 5.3801500000000002E-01
    lgam = [ 1.0000000000000000E+00, 1.1056357999999999E+00 ]
    gam3 = 1.5000000000000002E-01
    zeff = 4.2310780000000001E+00
    arep = 1.2476550000000000E+00
    xbond = 0.0000000000000000E+00
    dkernel = -4.1167399999999998E-03
    qkernel = 2.1358300000000000E-03
    mprad = 3.0000000000000000E+00
    mpvcn = 3.0000000000000000E+00
    en = 2.5499999999999998E+00
    """

    par = Param(**toml.loads(data))

    assert "H" in par.element
    assert "C" in par.element


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_param_calculator(dtype: torch.dtype) -> None:
    dd: DD = {"device": DEVICE, "dtype": dtype}
    numbers = symbol_to_number(["H", "C"])
    calc = Calculator(numbers, GFN1_XTB, opts={"verbosity": 0}, **dd)

    ref = torch.tensor([1.0, 4.0], **dd)

    h = calc.integrals.hcore
    assert h is not None

    occ = calc.ihelp.reduce_shell_to_atom(h.refocc)
    assert pytest.approx(ref.cpu()) == occ.cpu()


def _validate_param(par: Param) -> None:
    assert isinstance(par.meta, Meta)
    assert par.meta.name is not None

    if par.meta.name.casefold() == "gfn1-xtb":
        ref = GFN1_XTB
    elif par.meta.name.casefold() == "gfn2-xtb":
        ref = GFN2_XTB
    else:
        raise ValueError(f"Unknown parameter set: {par.meta.name}")

    for f, f_read in zip(ref.model_fields.keys(), par.model_fields.keys()):
        val = getattr(ref, f)
        val_read = getattr(par, f_read)
        assert val == val_read

    # GFN1_XTB is not really of type `Param`, but a `LazyLoaderParam`
    assert par == ref._loaded  # type: ignore


@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
def test_read_toml(parname: str) -> None:
    p = (
        Path(__file__).parents[2]
        / "src"
        / "dxtb"
        / "_src"
        / "param"
        / parname.split("-")[0]
    )
    par = Param.from_file(Path(p) / f"{parname}.toml")

    _validate_param(par)


@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
@pytest.mark.parametrize("ftype", ["json", "yaml"])
def test_read_other(ftype: str, parname: str) -> None:
    p = Path(__file__).parent / "param"
    par = Param.from_file(Path(p) / f"{parname}.{ftype}")

    _validate_param(par)


@pytest.mark.skipif(toml_w is None, reason="No TOML writer installed.")
@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
def test_write_toml(parname: str) -> None:
    p = (
        Path(__file__).parents[2]
        / "src"
        / "dxtb"
        / "_src"
        / "param"
        / parname.split("-")[0]
    )
    par = Param.from_file(p / f"{parname}.toml")

    with td.TemporaryDirectory() as tmp:
        # write to file
        p_write = Path(tmp) / "test.toml"
        par.to_file(p_write)

        # read the written file
        par_read = Param.from_file(p_write)

        _validate_param(par_read)


@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
def test_write_json(parname: str) -> None:
    ftype = "json"
    p = Path(__file__).parent / "param"
    par = Param.from_file(p / f"{parname}.{ftype}")

    with td.TemporaryDirectory() as tmp:
        # write to file
        p_write = Path(tmp) / f"test.{ftype}"
        par.to_file(p_write)

        # read the written file
        par_read = Param.from_file(p_write)

        _validate_param(par_read)


@pytest.mark.skipif(yaml is None, reason="No YAML writer installed.")
@pytest.mark.parametrize("parname", ["gfn1-xtb", "gfn2-xtb"])
@pytest.mark.parametrize("ftype", ["yml", "yaml"])
def test_write_yaml(ftype: str, parname: str) -> None:
    p = Path(__file__).parent / "param"
    par = Param.from_file(p / f"{parname}.yaml")

    with td.TemporaryDirectory() as tmp:
        # write to file
        p_write = Path(tmp) / f"test.{ftype}"
        par.to_file(p_write)

        # read the written file
        par_read = Param.from_file(p_write)

        _validate_param(par_read)
