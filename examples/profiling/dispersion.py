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
import torch

import dxtb
from dxtb._src.typing import DD

dd: DD = {"device": torch.device("cuda:0"), "dtype": torch.double}

NREPEATS = 100
n = 6000
chunk_size = 200
batch_mode = 0

print(f"Running on {dd['device']} with {n} atoms.")


numbers = torch.randint(1, 86, (n,), device=dd["device"])
positions = torch.rand((n, 3), **dd) * 10

# warmup for CUDA
if dd["device"] is not None:
    if dd["device"].type == "cuda":
        _ = torch.rand(100, 100, **dd)
        del _

######################################################################

dxtb.timer.reset()
dxtb.timer.start("Setup")

dxtb.timer.start("Ihelp", parent_uid="Setup")
ihelp = dxtb.IndexHelper.from_numbers(numbers, dxtb.GFN1_XTB, batch_mode=batch_mode)
dxtb.timer.stop("Ihelp")

dxtb.timer.start("Class", parent_uid="Setup")
obj = dxtb.new_dispersion(numbers, dxtb.GFN1_XTB, **dd)
assert obj is not None
dxtb.timer.stop("Class")

dxtb.timer.stop("Setup")
dxtb.timer.start("Cache")

torch.cuda.synchronize()
cache = obj.get_cache(numbers, ihelp=ihelp)

torch.cuda.synchronize()
dxtb.timer.stop("Cache")
torch.cuda.synchronize()
dxtb.timer.start("Energy")

e = obj.get_energy(positions, cache, chunk_size=chunk_size)
torch.cuda.synchronize()
dxtb.timer.stop("Energy")

dxtb.timer.print(v=1)


######################################################################

numbers = numbers.cpu()
positions = positions.cpu()
ihelp = ihelp.cpu()
dd: DD = {"device": torch.device("cpu"), "dtype": torch.double}

dxtb.timer.reset()
dxtb.timer.start("Setup")

dxtb.timer.start("Ihelp", parent_uid="Setup")
ihelp = dxtb.IndexHelper.from_numbers(numbers, dxtb.GFN1_XTB, batch_mode=batch_mode)
dxtb.timer.stop("Ihelp")

dxtb.timer.start("Class", parent_uid="Setup")
obj = dxtb.new_dispersion(numbers, dxtb.GFN1_XTB, **dd)
assert obj is not None
dxtb.timer.stop("Class")

dxtb.timer.stop("Setup")
dxtb.timer.start("Cache")

cache = obj.get_cache(numbers, ihelp=ihelp)

dxtb.timer.stop("Cache")
dxtb.timer.start("Energy")

e2 = obj.get_energy(positions, cache, chunk_size=chunk_size)

dxtb.timer.stop("Energy")

dxtb.timer.print(v=1)

print(e.sum())
print(e2.sum())
