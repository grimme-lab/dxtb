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
Data for SCF energies.
"""

from __future__ import annotations

import torch

from dxtb._src.typing import Molecule, Tensor, TypedDict

from ..molecules import merge_nested_dicts, mols


class Refs(TypedDict):
    """Format of reference records containing GFN1-xTB and GFN2-xTB reference values."""

    escf: Tensor
    """SCF energy for GFN1-xTB"""

    hessian: Tensor
    """
    SCF Hessian for GFN1-xTB. Must be reshaped using
    `reshape_fortran(x, torch.Size(2*(numbers.shape[-1], 3))`.
    """


class Record(Molecule, Refs):
    """Store for molecular information and reference values"""


refs: dict[str, Refs] = {
    "H": {
        "escf": torch.tensor(-4.0142947446183e-01, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "C": {
        "escf": torch.tensor(-1.7411359557542, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "Rn": {
        "escf": torch.tensor(-3.6081562853046, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "H2": {
        "escf": torch.tensor(-1.0585984032484, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "LiH": {
        "escf": torch.tensor(-0.88306406116865, dtype=torch.float64),
        "hessian": torch.tensor(
            [
                [+6.30114301276044e-03, -6.30114301276044e-03, +0.00000000000000e00],
                [+0.00000000000000e00, +0.00000000000000e00, +0.00000000000000e00],
                [-6.30114301276044e-03, +6.30114301276044e-03, +0.00000000000000e00],
                [+0.00000000000000e00, +0.00000000000000e00, +0.00000000000000e00],
                [+0.00000000000000e00, +0.00000000000000e00, +6.30114301240897e-03],
                [-6.30114301240897e-03, +0.00000000000000e00, +0.00000000000000e00],
                [+0.00000000000000e00, +0.00000000000000e00, -6.30114301240897e-03],
                [+6.30114301240897e-03, +0.00000000000000e00, +0.00000000000000e00],
                [-5.78546453009199e-13, +5.78546453009199e-13, +0.00000000000000e00],
                [+0.00000000000000e00, +1.45368268635385e-02, -1.45368268635211e-02],
                [+5.78546453009199e-13, -5.78546453009199e-13, +0.00000000000000e00],
                [+0.00000000000000e00, -1.45368268635385e-02, +1.45368268635211e-02],
            ],
            dtype=torch.float64,
        ).flatten(),
    },
    "HLi": {
        "escf": torch.tensor(-0.88306406116865, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "HC": {
        "escf": torch.tensor(0.0, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "HHe": {
        "escf": torch.tensor(0.0, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "S2": {
        "escf": torch.tensor(-7.3285116888517, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "H2O": {
        "escf": torch.tensor(-5.8052489623704e00, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "CH4": {
        "escf": torch.tensor(-4.3393059719255e00, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "SiH4": {
        "escf": torch.tensor(-4.0384093532453, dtype=torch.float64),
        "hessian": torch.tensor(
            [
                [+1.08640112902982e-01, -2.71600282261697e-02, -2.71600282253544e-02],
                [-2.71600282262911e-02, -2.71600282251289e-02, +4.19586240751890e-13],
                [+1.80729265421287e-03, +1.80729265506288e-03, -1.80729265426491e-03],
                [-1.80729265539248e-03, +2.84060969191202e-13, -1.80729265455981e-03],
                [+1.80729265516696e-03, +1.80729265436899e-03, -1.80729265537513e-03],
                [-2.71600282246269e-02, +3.20747426166973e-02, -3.49874195008232e-03],
                [+2.08276950826661e-03, -3.49874195020375e-03, +1.80739650643591e-03],
                [+3.03478750192376e-03, -4.84213208359202e-03, +3.86394467120216e-03],
                [-3.86399659595138e-03, -1.80739650614751e-03, -3.03478750190642e-03],
                [+3.86399659605546e-03, -3.86394467121950e-03, +4.84213208322773e-03],
                [-2.71600282248123e-02, -3.49874194985680e-03, +3.20747426170616e-02],
                [-3.49874195056804e-03, +2.08276950819722e-03, +1.80739650457596e-03],
                [-4.84213208277670e-03, +3.03478750228806e-03, -3.86399659515341e-03],
                [+3.86394467108073e-03, +1.80739650616622e-03, -3.86399659605546e-03],
                [+3.03478750204519e-03, -4.84213208324508e-03, +3.86394467113277e-03],
                [-2.71600282244564e-02, +2.08276950783293e-03, -3.49874195023844e-03],
                [+3.20747426170616e-02, -3.49874195016905e-03, -1.80739650553142e-03],
                [-3.86394467121950e-03, +3.86399659569117e-03, -3.03478750178499e-03],
                [+4.84213208279405e-03, +1.80739650568076e-03, +3.86394467135828e-03],
                [-4.84213208268996e-03, +3.03478750195846e-03, -3.86399659647180e-03],
                [-2.71600282249993e-02, -3.49874194996089e-03, +2.08276950807579e-03],
                [-3.49874194987415e-03, +3.20747426168708e-02, -1.80739650634728e-03],
                [+3.86399659615955e-03, -3.86394467092460e-03, +4.84213208286344e-03],
                [-3.03478750171560e-03, -1.80739650697286e-03, +4.84213208319303e-03],
                [-3.86394467095930e-03, +3.86399659635037e-03, -3.03478750159417e-03],
                [-1.96782694306119e-13, +1.80729265429960e-03, +1.80729265514962e-03],
                [-1.80729265449042e-03, -1.80729265478533e-03, +1.08640112903167e-01],
                [-2.71600282259442e-02, -2.71600282256319e-02, -2.71600282259615e-02],
                [-2.71600282257013e-02, -6.86571025726446e-13, -1.80729265438634e-03],
                [+1.80729265483737e-03, -1.80729265471594e-03, +1.80729265494145e-03],
                [+1.80739650574799e-03, +3.03478750199315e-03, -4.84213208288078e-03],
                [-3.86399659591669e-03, +3.86394467111542e-03, -2.71600282257013e-02],
                [+3.20747426169055e-02, -3.49874194945782e-03, -3.49874194971803e-03],
                [+2.08276950795436e-03, -1.80739650635758e-03, -3.03478750187172e-03],
                [+3.86399659610751e-03, +4.84213208305426e-03, -3.86394467101134e-03],
                [+1.80739650554009e-03, -4.84213208274201e-03, +3.03478750181968e-03],
                [+3.86394467172257e-03, -3.86399659638506e-03, -2.71600282242328e-02],
                [-3.49874194992619e-03, +3.20747426163850e-02, +2.08276950764211e-03],
                [-3.49874194977007e-03, +1.80739650536851e-03, -3.86399659595138e-03],
                [+3.03478750188907e-03, +3.86394467153175e-03, -4.84213208268996e-03],
                [-1.80739650533111e-03, +3.86399659612485e-03, -3.86394467153175e-03],
                [-3.03478750181968e-03, +4.84213208262058e-03, -2.71600282252129e-02],
                [-3.49874195004762e-03, +2.08276950821457e-03, +3.20747426168708e-02],
                [-3.49874194975272e-03, -1.80739650746021e-03, +4.84213208317569e-03],
                [-3.86394467092460e-03, -3.03478750159417e-03, +3.86399659680139e-03],
                [-1.80739650537854e-03, -3.86394467151441e-03, +3.86399659619424e-03],
                [+4.84213208256853e-03, -3.03478750185437e-03, -2.71600282254221e-02],
                [+2.08276950788497e-03, -3.49874194985680e-03, -3.49874194942312e-03],
                [+3.20747426169055e-02, +1.80739650663107e-03, +3.86394467113277e-03],
                [-4.84213208293283e-03, -3.86399659657588e-03, +3.03478750171560e-03],
                [-2.31748214368777e-13, -1.80729265455981e-03, +1.80729265455981e-03],
                [+1.80729265455981e-03, -1.80729265423021e-03, +2.09522069832824e-13],
                [-1.80729265473328e-03, +1.80729265506288e-03, -1.80729265485471e-03],
                [+1.80729265433430e-03, +1.08640112903843e-01, -2.71600282258228e-02],
                [-2.71600282257534e-02, -2.71600282258921e-02, -2.71600282263779e-02],
                [-1.80739650636137e-03, -3.03478750197581e-03, -3.86394467082052e-03],
                [+3.86399659576056e-03, +4.84213208331447e-03, -1.80739650484024e-03],
                [-3.03478750209724e-03, -3.86394467149706e-03, +4.84213208241241e-03],
                [+3.86399659609016e-03, -2.71600282246635e-02, +3.20747426170269e-02],
                [+2.08276950804109e-03, -3.49874195034253e-03, -3.49874195003028e-03],
                [+1.80739650537204e-03, +3.86394467102869e-03, +3.03478750232275e-03],
                [-4.84213208307160e-03, -3.86399659565648e-03, +1.80739650729352e-03],
                [+3.86394467045623e-03, +3.03478750161151e-03, -3.86399659593403e-03],
                [-4.84213208343590e-03, -2.71600282238403e-02, +2.08276950809314e-03],
                [+3.20747426169402e-02, -3.49874195041192e-03, -3.49874195077621e-03],
                [+1.80739650594395e-03, -3.86399659586464e-03, -4.84213208312365e-03],
                [+3.03478750213193e-03, +3.86394467095930e-03, -1.80739650626542e-03],
                [+4.84213208331447e-03, +3.86399659603812e-03, -3.03478750207989e-03],
                [-3.86394467101134e-03, -2.71600282240837e-02, -3.49874195074151e-03],
                [-3.49874195018640e-03, +3.20747426169055e-02, +2.08276950798905e-03],
                [-1.80739650630310e-03, +4.84213208327977e-03, +3.86399659612485e-03],
                [-3.86394467134094e-03, -3.03478750169825e-03, +1.80739650651208e-03],
                [-3.86399659622894e-03, -4.84213208310630e-03, +3.86394467102869e-03],
                [+3.03478750173294e-03, -2.71600282242268e-02, -3.49874195025579e-03],
                [-3.49874195027314e-03, +2.08276950807579e-03, +3.20747426166973e-02],
            ],
            dtype=torch.double,
        ).flatten(),
    },
    "PbH4-BiH3": {
        "escf": torch.tensor(-7.6074262079844, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "C6H5I-CH3SH": {
        "escf": torch.tensor(-27.612142805843, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "MB16_43_01": {
        "escf": torch.tensor(-33.200116717478, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "LYS_xao": {
        "escf": torch.tensor(-48.850798066902, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "LYS_xao_dist": {
        "escf": torch.tensor(-47.020544162958, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "C60": {
        "escf": torch.tensor(-128.79148324775, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
    "vancoh2": {
        "escf": torch.tensor(-3.2618651888175e02, dtype=torch.float64),
        "hessian": torch.tensor([], dtype=torch.float64),
    },
}


samples: dict[str, Record] = merge_nested_dicts(mols, refs)
