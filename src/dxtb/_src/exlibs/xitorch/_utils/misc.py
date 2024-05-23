# This file is part of dxtb, modified from xitorch/xitorch.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Original file licensed under the MIT License by xitorch/xitorch.
# Modifications made by Grimme Group.
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
import contextlib
import copy
from typing import Callable, Dict, List, Mapping, Union

import torch

__all__ = [
    "set_default_option",
    "get_and_pop_keys",
    "get_method",
    "dummy_context_manager",
    "TensorNonTensorSeparator",
    "TensorPacker",
]


def set_default_option(defopt: Dict, opt: Dict) -> Dict:
    # return a dictionary based on the options and if no item from option,
    # take it from defopt

    # make a shallow copy to detach the results from defopt
    res = copy.copy(defopt)
    res.update(opt)
    return res


def get_and_pop_keys(dct: Dict, keys: List) -> Dict:
    res = {}
    for k in keys:
        res[k] = dct.pop(k)
    return res


def get_method(
    algname: str, methods: Mapping[str, Callable], method: Union[str, Callable]
) -> Callable:
    if isinstance(method, str):
        methodname = method.lower()
        if methodname in methods:
            return methods[methodname]
        else:
            raise RuntimeError(f"Unknown {algname} method: {method}")
    elif hasattr(method, "__call__"):
        return method
    elif method is None:
        assert False, (
            "Internal assert failed, method in get_method is not supposed"
            "to be None. If this shows, then the corresponding function fails to "
            "set the default method"
        )
    else:
        raise TypeError(
            "Invalid method type: %s. Only str and callable are accepted."
            % type(method)
        )


@contextlib.contextmanager
def dummy_context_manager():
    yield None


class TensorNonTensorSeparator:
    """
    Class that provides function to separate/combine tensors and nontensors
    parameters.
    """

    def __init__(self, params, varonly=True):
        """
        Params is a list of tensor or non-tensor to be splitted into
        tensor/non-tensor
        """
        self.tensor_idxs = []
        self.tensor_params = []
        self.nontensor_idxs = []
        self.nontensor_params = []
        self.nparams = len(params)
        for i, p in enumerate(params):
            if isinstance(p, torch.Tensor) and (
                (varonly and p.requires_grad) or (not varonly)
            ):
                self.tensor_idxs.append(i)
                self.tensor_params.append(p)
            else:
                self.nontensor_idxs.append(i)
                self.nontensor_params.append(p)
        self.alltensors = len(self.tensor_idxs) == self.nparams

    def get_tensor_params(self):
        return self.tensor_params

    def ntensors(self):
        return len(self.tensor_idxs)

    def nnontensors(self):
        return len(self.nontensor_idxs)

    def reconstruct_params(self, tensor_params, nontensor_params=None):
        if nontensor_params is None:
            nontensor_params = self.nontensor_params
        if len(tensor_params) + len(nontensor_params) != self.nparams:
            raise ValueError(
                "The total length of tensor and nontensor params "
                "do not match with the expected length: %d instead of %d"
                % (len(tensor_params) + len(nontensor_params), self.nparams)
            )
        if self.alltensors:
            return tensor_params

        params = [None for _ in range(self.nparams)]
        for nidx, p in zip(self.nontensor_idxs, nontensor_params):
            params[nidx] = p
        for idx, p in zip(self.tensor_idxs, tensor_params):
            params[idx] = p
        return params


class TensorPacker:
    def __init__(self, tensors):
        self.idx_shapes = []
        istart = 0
        for i, p in enumerate(tensors):
            ifinish = istart + torch.numel(p)
            self.idx_shapes.append((istart, ifinish, p.shape))
            istart = ifinish

    def flatten(self, y_list):
        return torch.cat([y.reshape(-1) for y in y_list], dim=-1)

    def pack(self, y):
        yshapem1 = y.shape[:-1]
        return tuple(
            y[..., istart:ifinish].reshape(yshapem1 + shape)
            for (istart, ifinish, shape) in self.idx_shapes
        )
