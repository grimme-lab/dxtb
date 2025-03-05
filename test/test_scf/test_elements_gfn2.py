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
Test for SCF.
Reference values obtained with tblite 0.2.1 disabling repulsion and dispersion.

Note: Spin can be explicitly passed through options but it also works by letting
the corresponding function figure out the alpha/beta occupation automatically.

Atoms generally converged rather badly. Hence, the threshold for comparison is
significantly lower in this test suite. Additionally, I found that converging
the potential works better than converging the charges (`scp_mode="potential"`).
If the charges are converged, the orbtial-resolved charges usually stray quite
far away from zero. Nevertheless, the energy often oscillates anyway.
"""

from __future__ import annotations

import pytest
import torch
from tad_mctc.batch import pack

from dxtb import GFN2_XTB, Calculator
from dxtb._src.constants import labels
from dxtb._src.typing import DD

from ..conftest import DEVICE
from .samples import samples
from .uhf_table import uhf, uhf_anion, uhf_cation

# fmt: off
ref = torch.tensor(
    [
        -0.3934827590437188,-1.7431266329458670,-0.1800716865751749,
        -0.5691059816155888,-0.9524366145568731,-1.7951105194038208,
        -2.6094524546320614,-3.7694210954143372,-4.6193399554659953,
        -5.9322150527578561,-0.1670967498216340,-0.4659746637924263,
        -0.9053286117929751,-1.5714240856447492,-2.3778070880856061,
        -3.1482710158768512,-4.4825251342921133,-4.2790432675899392,
        -0.1657522390614218,-0.3716463524887721,-0.8541832932455158,
        -1.3670573060841023,-1.7181251908054325,-1.7474975140354929,
        -2.6191360738389711,-2.9467954760610584,-3.5067898492173386,
        -4.6683274353639224,-3.7480061309852375,-0.5275214022962790,
        -1.0811118359513132,-1.8099037845688559,-2.2394259485957670,
        -3.1204361961032472,-4.0483393705697406,-4.2718555408483345,
        -0.1599989486753450,-0.4624308530006268,-1.1948528971312189,
        -1.3106537576623940,-1.7812008396036578,-1.7846204189006885,
        -2.4820150859709535,-3.0202741319471276,-3.8958154714151534,
        -4.4098452999301987,-3.8217382102713393,-0.5330372553013433,
        -1.1259377791295933,-2.0128966169144547,-2.1642287880347792,
        -3.0090909635721594,-3.7796302627467928,-3.8835884981900253,
        -0.1485299624614294,-0.4336420207320540,-1.2047930237499953,
        -0.8995088859104427,-0.8933692906260793,-0.8872296591025184,
        -0.8810900618954706,-0.8749504289172132,-0.8688108329556412,
        -0.8626712738164258,-0.8565316211158370,-0.8503920620019818,
        -0.8442524661402829,-0.8381128335296505,-0.8319732376684492,
        -0.8258335683086163,-0.8196940091967588,-1.3119987657554162,
        -1.9048135505602584,-2.2150777726130775,-3.0080349133279856,
        -3.0705864555234426,-3.6450039004282027,-4.4374585721609048,
        -3.8026194480681164,-0.8480322467084417,-1.4386851451418587,
        -2.2048409808696365,-2.2666534163614425,-2.7347817366291736,
        -3.0005430918001297,-3.8578865356212368
    ]
)


ref_cation = torch.tensor(
    [
        +0.2295521666666667,-0.4838821498062669, 0.1659637000000000,
        +0.0382879603144248,-0.3028735733744940,-0.8032323515803012,
        -1.9127945803017519,-2.9485380631567804,-3.7083708531554338,
        -4.7615133525389206, 0.1954855666666667, 0.0177291347704535,
        -0.4909477993783444,-1.1519234984400135,-1.8635041144653153,
        -2.5869831371080387,-3.8074087383765951,-3.5353072019821710,
        +0.1915705000000000, 0.0412624237556138,-0.4876393461395803,
        -1.0512553851526367,-1.2276387992117728,-1.3603969166047660,
        -2.1531486571035261,-2.6154080943715741,-2.9630475933202085,
        -4.1356689668589270,-3.3754409391861020, 0.0954118321851938,
        -0.7770718988731139,-1.3380776245766277,-1.7274789356622700,
        -2.5727065347449729,-3.3672645851873964,-3.6095838966360381,
        +0.2024472666666667,-0.0012154265003135,-0.8205283144950449,
        -0.9268352102582531,-1.1831321107080399,-1.3548034028896174,
        -2.0946916383295942,-2.4392774466851530,-3.3116283806720501,
        -3.8839746929916270,-3.3861214800526964,-0.0014053943173384,
        -0.8313996799454058,-1.5499915918867848,-1.7004451789474953,
        -2.3281239391233317,-3.1287586760341632,-3.3107276971589097,
        +0.1984240000000000,-0.0248826103660272,-0.8043328582495408,
        -0.5782798242579857,-0.5658520874417623,-0.5534913129723908,
        -0.5411976806647678,-0.5289709450687978,-0.5168113516345733,
        -0.5047187940458501,-0.4926934202118626,-0.4807349495874395,
        -0.4688434826446651,-0.4570190103215450,-0.4452616792885216,
        -0.4335712785474503,-0.4219479871505273,-0.8307280647116214,
        -1.4308333086427949,-1.5144142097252595,-2.4712730319839222,
        -2.6040096866434608,-3.1254832081174122,-3.8878107594341054,
        -3.2726654766627710,-0.1990876900208875,-1.2218139762625606,
        -1.8331228693386723,-1.8241019780597743,-2.2952188045868103,
        -2.5146156635502148,-3.3252215178719071
    ]
)


ref_anion = torch.tensor(
    [
        -0.6107466928548624,-1.5242281622618639,-0.2811010731503498,
        -0.0225472225337581,-0.8833252789733284,-1.9209221941523817,
        -2.8228473813671178,-4.0689442489122936,-4.9096355171858255,
        -5.8525351484414445,-0.2586230663099347,-0.3777253521591805,
        -0.9769314233128334,-1.5740591001415243,-2.6051960359906325,
        -3.4046900452338020,-4.7851339537609663,-4.0730096623488325,
        -0.2754729781228437,-0.2991556954084400,-0.8448287925548863,
        -1.4204576650727849,-1.6417599547371493,-1.7349725353358738,
        -2.7795518977731328,-2.8451293928733801,-3.7199033615164447,
        -4.7200304836864646,-3.9176023227843726,-0.3029556119060911,
        -1.1870387794933592,-1.7866119117748076,-2.3590107738630417,
        -3.4032788583401246,-4.3322311664719511,-4.1134660808242316,
        -0.3119641640173567,-0.3850328498442683,-1.0956747182030877,
        -1.4355530563288270,-1.9640589777716178,-2.2204613546902952,
        -2.4156522277327448,-3.0555901876944822,-4.0966362950940773,
        -4.5493845820943051,-3.9936149404906232,-0.3539529575986469,
        -1.2296302217900721,-2.0597413560462923,-2.2773377812690265,
        -3.0548938936900618,-4.0603555990360469,-3.7399393131122793,
        -0.2589149249228587,-0.4142473193609475,-1.3583939757463996,
        -1.1939756095103720,-1.1787219836159635,-1.1634684731233553,
        -1.1482147512203462,-1.1329610795197471,-1.1177073276791298,
        -1.1024536912403247,-1.0871998378275738,-1.0719461382994526,
        -1.0566921636869997,-1.0414384012947371,-1.0261844283637402,
        -1.0109304558921322,-0.9956765988223176,-1.4562264589533362,
        -2.0816650782323638,-2.5504231469608891,-3.1722176794464403,
        -3.2020176312576272,-3.6596592185939323,-4.5670020773987314,
        -3.8940033683322537,-0.8351587238649311,-1.5354913944230826,
        -2.2747911870200199,-2.3492048567342909,-3.1176094241608174,
        -3.2707609413543071,-3.7962721880829253
    ]
)
# fmt: on


opts = {
    "fermi_etemp": 300,
    "fermi_maxiter": 500,
    "scf_mode": labels.SCF_MODE_IMPLICIT_NON_PURE,
    "scp_mode": labels.SCP_MODE_FOCK,  # better convergence for atoms
    "verbosity": 0,
}


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2  # math.sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    numbers = torch.tensor([number], device=DEVICE)
    positions = torch.zeros((1, 3), **dd)
    r = ref[number - 1]
    charges = torch.tensor(0.0, **dd)

    # opts["spin"] = uhf[number - 1]
    atol = 1e-5 if dtype == torch.float else 1e-6
    options = dict(
        opts,
        **{
            "f_atol": atol,
            "x_atol": atol,
            "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
        },
    )
    calc = Calculator(numbers, GFN2_XTB, opts=options, **dd)
    results = calc.singlepoint(positions, charges)
    assert pytest.approx(r.cpu(), abs=tol) == results.scf.sum(-1).cpu()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_cation(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # SCF does not converge (in tblite too)
    if number in (5, 6):
        return

    if number == 18:
        tol = 5e-2

    numbers = torch.tensor([number], device=DEVICE)
    positions = torch.zeros((1, 3), **dd)
    r = ref_cation[number - 1]
    charges = torch.tensor(1.0, **dd)
    spin = uhf_cation[number - 1]

    options = dict(
        opts,
        **{
            "f_atol": 1e-5,  # avoids Jacobian inversion error
            "x_atol": 1e-5,  # avoids Jacobian inversion error
            "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
            "scp_mode": labels.SCP_MODE_FOCK,
        },
    )
    calc = Calculator(numbers, GFN2_XTB, opts=options, **dd)

    # no (valence) electrons: [1, 3, 11, 19, 37, 55]
    results = calc.singlepoint(positions, charges, spin=spin)
    assert pytest.approx(r.cpu(), abs=tol) == results.scf.sum(-1).cpu()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_anion(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2  # math.sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": DEVICE, "dtype": dtype}

    # Helium doesn't have enough orbitals for negative charge
    if number == 2:
        return

    # SCF does not converge (in tblite too)
    if number in [24]:
        tol = 1e-1

    numbers = torch.tensor([number], device=DEVICE)
    positions = torch.zeros((1, 3), **dd)
    r = ref_anion[number - 1]
    charges = torch.tensor(-1.0, **dd)
    spin = uhf_anion[number - 1]

    options = dict(
        opts,
        **{
            "f_atol": 1e-5,  # avoid Jacobian inversion error
            "x_atol": 1e-5,  # avoid Jacobian inversion error
            "fermi_thresh": 1e-4 if dtype == torch.float32 else 1e-10,
        },
    )
    calc = Calculator(numbers, GFN2_XTB, opts=options, **dd)
    results = calc.singlepoint(positions, charges, spin=spin)

    assert pytest.approx(r.cpu(), abs=tol) == results.scf.sum(-1).cpu()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", [1, 2, 10, 25, 26, 50, 86])
@pytest.mark.parametrize("mol", ["SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_batch(dtype: torch.dtype, number: int, mol: str) -> None:
    tol = 1e-1
    dd: DD = {"device": DEVICE, "dtype": dtype}

    sample = samples[mol]
    numbers = pack(
        (
            sample["numbers"].to(device=DEVICE),
            torch.tensor([number], device=DEVICE),
        )
    )
    positions = pack(
        (
            sample["positions"].to(**dd),
            torch.zeros((1, 3), **dd),
        ),
    )
    refs = pack(
        (
            sample["egfn2"].to(**dd),
            ref[number - 1].to(**dd),
        )
    )
    charges = torch.tensor([0.0, 0.0], **dd)
    spin = torch.tensor([0, uhf[number - 1]])

    calc = Calculator(numbers, GFN2_XTB, opts=opts, **dd)
    results = calc.singlepoint(positions, charges, spin=spin)

    assert pytest.approx(refs.cpu(), abs=tol) == results.scf.sum(-1).cpu()
