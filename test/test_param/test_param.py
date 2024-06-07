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

from __future__ import annotations

import pytest
import tomli as toml
import torch
from tad_mctc.convert import symbol_to_number

from dxtb._src.param.meta import Meta
from dxtb._src.typing import DD

from ..conftest import DEVICE


def test_builtin_gfn1() -> None:
    # pylint: disable=import-outside-toplevel
    from dxtb._src.param.gfn1 import GFN1_XTB as par

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
    # pylint: disable=import-outside-toplevel
    from dxtb._src.param import Param

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
    # pylint: disable=import-outside-toplevel
    from dxtb import Calculator
    from dxtb._src.param.gfn1 import GFN1_XTB as par

    dd: DD = {"device": DEVICE, "dtype": dtype}
    numbers = symbol_to_number(["H", "C"])
    calc = Calculator(numbers, par, opts={"verbosity": 0}, **dd)

    ref = torch.tensor([1.0, 4.0], **dd)

    h = calc.integrals.hcore
    assert h is not None

    occ = calc.ihelp.reduce_shell_to_atom(h.integral.refocc)
    assert pytest.approx(ref.cpu()) == occ.cpu()
