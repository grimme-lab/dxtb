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
import time

t0 = time.perf_counter()

############################

import torch

t1 = time.perf_counter()

############################

from dxtb import GFN1_XTB

t2 = time.perf_counter()

############################

_ = GFN1_XTB.model_copy()
t3 = time.perf_counter()

############################

import scipy

t4 = time.perf_counter()

############################

print("Torch", t1 - t0)
print("dxtb", t2 - t1)
print("Param", t3 - t2)
print("scipy", t4 - t3)

del scipy, torch, GFN1_XTB
