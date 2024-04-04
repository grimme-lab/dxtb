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

from dxtb.param import GFN1_XTB as par
from dxtb.typing import DD
from dxtb.utils import batch
from dxtb.xtb import Calculator

from .samples import samples

device = None

ref = torch.tensor(
    [
        -0.4014294744618301,
        -1.6258646654198721,
        -0.2671714426384922,
        -0.7012869049089436,
        -1.1998696279038878,
        -1.7411359557542052,
        -2.8988862104065958,
        -4.3526523408648030,
        -4.9969448424630265,
        -6.3190030086322286,
        -0.1733674706866860,
        -0.7328492086440124,
        -1.0775966895776903,
        -1.6252631326184159,
        -2.4250620380497385,
        -3.5359207433156339,
        -4.1663539440166675,
        -6.1469954466724239,
        -0.2137179709637750,
        -0.5864589388109813,
        -0.9739522227703182,
        -1.1487975500262675,
        -1.4032495694507467,
        -1.6258320924723875,
        -2.0432134739535242,
        -2.8490837602318742,
        -3.4039476610517951,
        -3.9783299398205854,
        -4.3908896126848882,
        -0.8278490035919727,
        -1.1559605434617155,
        -1.5172654693894492,
        -2.2030246081688953,
        -3.2227173551239399,
        -3.8180395534595282,
        -4.8421263290307968,
        -0.2799176883500828,
        -0.5027433177275922,
        -0.8559774488167702,
        -1.0015251619993044,
        -1.7072155954305537,
        -1.7786275624264398,
        -2.2160358545342351,
        -3.2181664696662833,
        -3.9147305338504088,
        -4.4068901473558819,
        -3.7633756826766938,
        -0.8892760127989151,
        -1.3780364568041852,
        -2.3412464596133091,
        -2.3522450904243550,
        -3.6750286938689936,
        -3.8857572752278440,
        -4.2321989780523870,
        -0.2330413901515095,
        -0.4742595783451311,
        -0.6959185846380349,
        -0.6485304192347468,
        -0.6450212630347002,
        -0.6415120701014334,
        -0.6380028404220172,
        -0.6344936474925504,
        -0.6309844913125303,
        -0.6274752983832106,
        -0.6239660687045753,
        -0.6204569125245851,
        -0.6169477195952713,
        -0.6134385634152814,
        -0.6099292969873201,
        -0.6064201408073303,
        -0.6029109478780164,
        -0.7596719486827229,
        -1.6934927181631920,
        -1.9903305781546048,
        -2.0068018142203417,
        -3.1552379430508148,
        -3.9539731772481859,
        -4.2489748260259601,
        -3.9019226424065825,
        -0.9152519783258882,
        -1.1137676526000495,
        -1.4989523256308206,
        -2.0838915690368061,
        -3.0676044864514225,
        -3.4437503367483560,
        -3.6081562853045592,
    ]
)

# fmt: off
uhf = torch.tensor(
    [
        1,                                                 0,
        1, 0,                               1, 0, 1, 0, 1, 0,
        1, 0,                               1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0
    ]
)
# fmt: on


ref_cation = torch.tensor(
    [
        0.2350495000000000,
        -0.0422428129239876,
        0.1369166666666667,
        -0.1836139857878052,
        -0.7021610492209647,
        -1.1060742863488979,
        -2.1782414865973068,
        -3.3929027848641802,
        -3.9486100517085290,
        -5.4089550104332629,
        0.1229540000000000,
        -0.1526824376553396,
        -0.7929165898783470,
        -1.2064302987580611,
        -1.7319786163576927,
        -2.7725673355423974,
        -3.5179743056411774,
        -5.2547187063374068,
        0.0816097666666667,
        -0.1889038694054907,
        -0.5164286712433095,
        -0.7054229863305306,
        -0.8953744943379764,
        -1.1145548860858483,
        -1.6781450836464422,
        -2.0768800363599724,
        -2.8098681418659432,
        -3.3918338006786093,
        -3.9272193399045281,
        -0.2625283351293197,
        -0.8517810578775546,
        -1.0787522669740286,
        -1.5017772126895934,
        -2.5547725893532198,
        -3.0519213538372423,
        -3.9953892543401519,
        0.0878675000000000,
        -0.1466078255304628,
        -0.4937299864801741,
        -0.6513197157489261,
        -1.2734962578300519,
        -1.3411315800579504,
        -1.7441817291584463,
        -2.6949495714667360,
        -3.3840208304086556,
        -3.8665790293167479,
        -3.2936887930479184,
        -0.3338044397327909,
        -1.1351439870281808,
        -1.9118351274462178,
        -1.8456782193519603,
        -3.0738678691656975,
        -3.2276435086854969,
        -3.5173994843193732,
        0.0925550000000000,
        -0.1182202891725656,
        -0.3383862439154567,
        -0.3380276085842653,
        -0.3364513307550568,
        -0.3350141427496482,
        -0.3337162359090419,
        -0.3325572206289337,
        -0.3315373115184752,
        -0.3306564922318164,
        -0.3299149902654105,
        -0.3293123596652018,
        -0.3288488188887932,
        -0.3285243679361848,
        -0.3283389200405289,
        -0.3282929953096204,
        -0.3283858232519615,
        -0.4538801172959733,
        -1.2190319743563898,
        -1.5399384468018016,
        -1.5831499553557236,
        -2.5862107373895995,
        -3.4087534399601136,
        -3.6246625551978040,
        -3.2948988109803317,
        -0.2990523224962774,
        -0.8284450281457056,
        -1.1284674720252441,
        -1.6724837041091658,
        -2.4123355716184571,
        -2.6886076809857715,
        -2.9576573558829384,
    ]
)

# fmt: off
uhf_cation = torch.tensor(
    [
        0,                                                 1,
        0, 1,                               0, 1, 0, 1, 0, 1,
        0, 1,                               0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
)
# fmt: on

ref_anion = torch.tensor(
    [
        -0.5678094489236601,
        0.0000000000000000,
        -0.4659175519436511,
        -0.7944690166480025,
        -1.3514075648859643,
        -1.9170116002810331,
        -3.1284233130873655,
        -4.7053860329689856,
        -5.3222970343111768,
        -6.0135919541021741,
        -0.3037809414057536,
        -0.6889153966301176,
        -1.1294568900412956,
        -1.8636984933817820,
        -2.5218710230846475,
        -3.7615667932865060,
        -4.5279482362587169,
        -5.9532003542728642,
        -0.3946877084813858,
        -0.6601217419378675,
        -1.1181158489120291,
        -1.3177666092815929,
        -1.2384479883964847,
        -1.6487377642904479,
        -0.5954698324087719,
        -2.6284270918459773,
        -3.5618386589525119,
        -4.1031863005505338,
        -4.5943678854652479,
        -0.9409945569463229,
        -1.3059077459899349,
        -1.7645198843278482,
        -2.1964765797004659,
        -3.4039351935495046,
        -3.9571135364820273,
        -4.6240117892188310,
        -0.5719678767001657,
        -0.6583170517916040,
        -1.0130429561259551,
        -1.1770073221613631,
        -1.9625368412303521,
        -2.0338624073866263,
        -2.1752623138381653,
        -3.3802910855544024,
        -4.0402591326658541,
        -4.4158570177448571,
        -3.7600845723053635,
        -1.0180004822365232,
        -1.4888020156580570,
        -2.4537671300397492,
        -2.3767986294370997,
        -3.8685840856051446,
        -4.0435926774613034,
        -3.9095090580155762,
        -0.4735277804823783,
        -0.6639055123686469,
        -0.8747426483751759,
        -0.7930037109846064,
        -0.7859031407166542,
        -0.7774600374514302,
        -0.7692826709429081,
        -0.7615788166990638,
        -0.7541577457962461,
        -0.7468810501730819,
        -0.7397437693038101,
        -0.7327453598812527,
        -0.7258859667911156,
        -0.7191657370235338,
        -0.7125844368143500,
        -0.7061427332686532,
        -0.6998397088976513,
        -0.8340395816028910,
        -1.9776974757621952,
        -2.1930497428504525,
        -2.0018467092139165,
        -3.3776836100161272,
        -4.0976008074736150,
        -4.3290532643937203,
        -4.0370809224168633,
        -0.9642956138466999,
        -1.2716085097941741,
        -1.6601446321465445,
        -2.2260645738865312,
        -3.0614995810384213,
        -3.3648843601107044,
        -3.2192830086977700,
    ]
)

# fmt: off
uhf_anion = torch.tensor(
    [
        0,                                                 1,
        0, 1,                               0, 1, 0, 1, 0, 1,
        0, 1,                               0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]
)
# fmt: on

opts = {
    "fermi_etemp": 300,
    "fermi_maxiter": 500,
    "fermi_thresh": {
        torch.float32: torch.tensor(1e-4, dtype=torch.float32),  # instead of 1e-5
        torch.float64: torch.tensor(1e-10, dtype=torch.float64),
    },
    "scp_mode": "potential",  # important for atoms (better convergence)
    "verbosity": 0,
}


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2  # math.sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    numbers = torch.tensor([number])
    positions = torch.zeros((1, 3), **dd)
    r = ref[number - 1].item()
    charges = torch.tensor(0.0, **dd)

    # opts["spin"] = uhf[number - 1]
    options = dict(opts, **{"f_atol": 1e-6, "x_atol": 1e-6})
    calc = Calculator(numbers, par, opts=options, **dd)
    results = calc.singlepoint(numbers, positions, charges)
    assert pytest.approx(r, abs=tol) == results.scf.sum(-1).item()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_cation(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2
    dd: DD = {"device": device, "dtype": dtype}

    # SCF does not converge for gold (in tblite too)
    if number == 79:
        return

    numbers = torch.tensor([number])
    positions = torch.zeros((1, 3), **dd)
    r = ref_cation[number - 1].item()
    charges = torch.tensor(1.0, **dd)
    spin = uhf_cation[number - 1]

    options = dict(
        opts,
        **{
            "f_atol": 1e-5,  # avoids Jacobian inversion error
            "x_atol": 1e-5,  # avoids Jacobian inversion error
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)

    # no (valence) electrons: [1, 3, 11, 19, 37, 55]
    results = calc.singlepoint(numbers, positions, charges, spin=spin)
    assert pytest.approx(r, abs=tol) == results.scf.sum(-1).item()


@pytest.mark.large
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", range(1, 87))
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_anion(dtype: torch.dtype, number: int) -> None:
    tol = 1e-2  # math.sqrt(torch.finfo(dtype).eps) * 10
    dd: DD = {"device": device, "dtype": dtype}

    # Helium doesn't have enough orbitals for negative charge
    if number == 2:
        return

    # SCF does not converge (in tblite too)
    if number in [21, 22, 23, 25, 43, 57, 58, 59]:
        return

    numbers = torch.tensor([number])
    positions = torch.zeros((1, 3), **dd)
    r = ref_anion[number - 1].item()
    charges = torch.tensor(-1.0, **dd)
    spin = uhf_anion[number - 1]

    options = dict(
        opts,
        **{
            "f_atol": 1e-5,  # avoid Jacobian inversion error
            "x_atol": 1e-5,  # avoid Jacobian inversion error
        },
    )
    calc = Calculator(numbers, par, opts=options, **dd)
    results = calc.singlepoint(numbers, positions, charges, spin=spin)

    assert pytest.approx(r, abs=tol) == results.scf.sum(-1).item()


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("number", [1, 2, 10, 25, 26, 50, 86])
@pytest.mark.parametrize("mol", ["SiH4"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_element_batch(dtype: torch.dtype, number: int, mol: str) -> None:
    tol = 1e-1
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[mol]
    numbers = batch.pack((sample["numbers"], torch.tensor([number])))
    positions = batch.pack((sample["positions"], torch.zeros((1, 3)))).type(dtype)
    refs = batch.pack((sample["escf"], ref[number - 1])).type(dtype)
    charges = torch.tensor([0.0, 0.0], **dd)
    spin = torch.tensor([0, uhf[number - 1]])

    calc = Calculator(numbers, par, opts=opts, **dd)
    results = calc.singlepoint(numbers, positions, charges, spin=spin)

    assert pytest.approx(refs, abs=tol) == results.scf.sum(-1)
