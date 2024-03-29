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

import torch
from tad_mctc.typing import DD

from dxtb.basis import IndexHelper
from dxtb.integral import Hamiltonian
from dxtb.param import GFN1_XTB as par

device = torch.device("cpu")
dd: DD = {"device": device, "dtype": torch.double}

numbers = torch.randint(1, 86, (50,))
ihelp = IndexHelper.from_numbers(numbers, par)
hcore = Hamiltonian(numbers, par, ihelp, **dd)


def time_function(func, repeats: int, *args, **kwargs) -> tuple[float, float]:
    """
    Time the execution of a function over a specified number of repeats.

    Parameters:
    - func: The function to time.
    - repeats: The number of times to execute the function.
    - *args: Positional arguments to pass to the function.
    - **kwargs: Keyword arguments to pass to the function.

    Returns:
    - Tuple containing the total execution time and average execution time.
    """
    total_time = 0.0

    for _ in range(repeats):
        start_time = time.time()
        func(*args, **kwargs)  # Execute the function with provided arguments
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / repeats

    print(f"Total {func.__name__}: {total_time:.4f} seconds")
    print(f"Average {func.__name__}: {average_time:.4f} seconds")
    return total_time, average_time


REPS = 5
time_function(hcore.integral._get_hscale, REPS)


a1 = hcore.integral._get_hscale()
print((a1 == a1.mT).all())
