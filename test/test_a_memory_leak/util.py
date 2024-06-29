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
Utility functions for checking memory leaks.

Taken from DQC.
"""

from __future__ import annotations

import gc

import torch

from dxtb.__version__ import __tversion__
from dxtb._src.typing import Callable, Generator, Literal, Tensor, overload


def garbage_collect() -> None:
    """
    Run the garbage collector.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _tensors_from_gc(gpu_only=False) -> Generator[Tensor, None, None]:
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                tensor = obj
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                tensor = obj.data
            else:
                continue

            if tensor.is_cuda or not gpu_only:
                yield tensor
        except Exception:  # nosec B112 pylint: disable=broad-exception-caught
            continue


@overload
def _get_tensor_memory(
    return_number_tensors: Literal[False] = False,
) -> float: ...


@overload
def _get_tensor_memory(
    return_number_tensors: Literal[True] = True,
) -> tuple[float, int]: ...


def _get_tensor_memory(
    return_number_tensors: bool = False,
) -> float | tuple[float, int]:
    """
    Obtain the total memory occupied by torch.Tensor in the garbage collector.

    Returns
    -------
    tuple[float, int]
        Memory in MiB and number of tensors.
    """

    # obtaining all the tensor objects from the garbage collector

    # iterate each tensor objects uniquely and calculate the total storage
    visited_data = set()
    total_mem = 0.0
    count = 0
    for tensor in _tensors_from_gc():
        if tensor.is_sparse:
            continue

        if __tversion__ < (2, 0, 0):
            storage = tensor.storage()
        else:
            storage = tensor.untyped_storage()

        # check if it has been visited
        data_ptr = storage.data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        # calculate the storage occupied
        numel = storage.size()
        elmt_size = storage.element_size()
        mem = numel * elmt_size / (1024 * 1024)  # in MiB

        total_mem += mem
        count += 1

    if return_number_tensors is True:
        return total_mem, count
    return total_mem


def _show_memsize(fcn, ntries: int = 10, gccollect: bool = False):
    # show the memory growth
    size0, num0 = _get_tensor_memory(return_number_tensors=True)

    for i in range(ntries):
        fcn()
        if gccollect:
            gc.collect()
        size, num = _get_tensor_memory(return_number_tensors=True)

        print(
            f"{i + 1:2d} iteration: {size - size0:.16f} MiB of {num-num0:d} addtional tensors"
        )


def has_memleak_tensor(
    fcn: Callable, repeats: int = 5, gccollect: bool = False
) -> bool:
    """
    Assert no tensor memory leak when calling the function.

    Arguments
    ---------
    fcn: Callable
        A function with no input and output to be checked.
    gccollect: bool
        If True, then manually apply ``gc.collect()`` after the function
        execution.

    Returns
    -------
    bool
        Whether there is a memory leak (``True``) or not (``False``).
    """
    size0, num0 = _get_tensor_memory(return_number_tensors=True)

    fcn()
    if gccollect:
        gc.collect()

    size, num = _get_tensor_memory(return_number_tensors=True)

    if size0 != size or num0 != num:
        _show_memsize(fcn, repeats, gccollect=gccollect)

    return size0 != size
