#!/usr/bin/env python3
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
Comparing sequential and batched execution.
"""
from pathlib import Path

import torch
from tad_mctc import read, read_chrg

import dxtb
from dxtb.typing import DD

dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}
dxtb.timer.cuda_sync = False

# read molecule from file
p = Path(__file__).parent

if "molecules" not in [x.name for x in p.iterdir()]:
    p = p.parent

f = p / "molecules" / "nicotine.xyz"


numbers, positions = read(f, **dd)
charge = read_chrg(f, **dd)

# create batched input
BATCH = 16
numbers = torch.stack([numbers] * BATCH)
positions = torch.stack([positions] * BATCH)
charge = torch.stack([charge] * BATCH)

# same molecule -> batched mode 2
obatch = {"verbosity": 0, "batch_mode": 2}
oseq = {"verbosity": 0, "batch_mode": 0}


def run_seq():
    res = []
    for i in range(BATCH):
        calc = dxtb.Calculator(numbers[i], dxtb.GFN1_XTB, opts=oseq, **dd)
        res.append(calc.energy(positions[i], chrg=charge[i]))

    return torch.stack(res)


def run_batch():
    calc = dxtb.Calculator(numbers, dxtb.GFN1_XTB, opts=obatch, **dd)
    return calc.get_energy(positions, chrg=charge)


print("i seq   batch")
print("-------------")

t_batch, t_seq = [], []
for n in range(10):
    print(n, end=" ", flush=True)

    # sequential
    dxtb.timer.reset()
    run_seq()
    dxtb.timer.stop_all()

    tseq = dxtb.timer.get_time("total")
    t_seq.append(tseq)
    print(f"{tseq:.2f}", end=" ", flush=True)

    # batched
    dxtb.timer.reset()
    run_batch()
    dxtb.timer.stop_all()

    tbatch = dxtb.timer.get_time("total")
    t_batch.append(tbatch)
    print(f" {tbatch:.2f}")

print("-------------")
print(f"  {torch.tensor(t_seq).mean().item():.2f}", end="")
print(f"  {torch.tensor(t_batch).mean().item():.2f}")
