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
Simple energy calculation.
"""
import functools
import logging
import traceback

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


def log_tensor_move(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]

        # Get tensor details
        tensor_id = id(self)
        tensor_shape = tuple(self.size())
        tensor_dtype = self.dtype
        tensor_device = self.device

        # Capture stack trace
        stack = "".join(traceback.format_stack(limit=4)[:-1])

        # Only log if the tensor is moved to a different device
        if tensor_device == device:
            return func(self, *args, **kwargs)

        logging.info(
            f"Tensor ID: {tensor_id}, Shape: {tensor_shape}, Dtype: {tensor_dtype}, "
            f"From Device: {tensor_device}, To Device: {device}, "
            f"Called from:\n{stack}"
        )

        return func(self, *args, **kwargs)

    return wrapper


def override_tensor_methods():
    tensor_methods_to_override = ["to", "cuda", "cpu"]

    for method_name in tensor_methods_to_override:
        original_method = getattr(torch.Tensor, method_name)
        decorated_method = log_tensor_move(original_method)
        setattr(torch.Tensor, method_name, decorated_method)


override_tensor_methods()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

import dxtb

dd = {"dtype": torch.double, "device": torch.device("cuda:0")}

# LiH
numbers = torch.tensor([3, 1], device=dd["device"])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], **dd)

# instantiate a calculator
opts = {"verbosity": 6}
calc = dxtb.calculators.GFN1Calculator(numbers, opts=opts, **dd)

# compute the energy
pos = positions.clone().requires_grad_(True)
energy = calc.get_energy(pos)
