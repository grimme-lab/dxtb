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


class MyCube(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        result = x**3
        # In regular PyTorch, if we had just run y = x ** 3, then the backward
        # pass computes dx = 3 * x ** 2. In this autograd.Function, we've done
        # that computation here in the forward pass instead.
        dx = 3 * x**2
        return result, dx

    @staticmethod
    def setup_context(ctx, inputs, output):
        (x,) = inputs
        result, dx = output
        ctx.save_for_backward(x, dx)

    @staticmethod
    def backward(ctx, grad_output, grad_dx):
        x, dx = ctx.saved_tensors
        # In order for the autograd.Function to work with higher-order
        # gradients, we must add the gradient contribution of `dx`.
        result = grad_output * dx + grad_dx * 6 * x
        return result


def my_cube(x):
    result, _ = MyCube.apply(x)  # type: ignore
    return result


x = torch.tensor([4, 2], dtype=torch.double)
ggx = torch.func.jacrev(torch.func.jacrev(my_cube))(x)
print(ggx)
print(ggx.sum(0))

from tad_mctc.autograd import hessian

print(hessian(my_cube, (x,), argnums=0))
