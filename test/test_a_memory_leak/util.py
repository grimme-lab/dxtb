"""
Utility functions for checking memory leaks.

Taken from DQC.
"""
from __future__ import annotations

from dxtb._types import Tensor, Callable
import gc


def _get_tensor_memory() -> float:
    """
    Obtain the total memory occupied by torch.Tensor in the garbage collector.

    Returns
    -------
    float
        Memory in MiB
    """

    # obtaining all the tensor objects from the garbage collector
    tensor_objs = [obj for obj in gc.get_objects() if isinstance(obj, Tensor)]

    # iterate each tensor objects uniquely and calculate the total storage
    visited_data = set([])
    total_mem = 0.0
    for tensor in tensor_objs:
        if tensor.is_sparse:
            continue

        # check if it has been visited
        storage = tensor.storage()
        data_ptr = storage.data_ptr()  # type: ignore
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        # calculate the storage occupied
        numel = storage.size()
        elmt_size = storage.element_size()
        mem = numel * elmt_size / (1024 * 1024)  # in MiB

        total_mem += mem

    return total_mem


def _show_memsize(fcn, ntries: int = 10, gccollect: bool = False):
    # show the memory growth
    size0 = _get_tensor_memory()
    for i in range(ntries):
        fcn()
        if gccollect:
            gc.collect()
        size = _get_tensor_memory()

        print(f"{i + 1:3d} iteration: {size - size0:.7f} MiB of tensors")


def has_memleak_tensor(fcn: Callable, gccollect: bool = False) -> bool:
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
        Whether there is a memory leak (`True`) or not (`False`).
    """
    size0 = _get_tensor_memory()
    fcn()
    if gccollect:
        gc.collect()
    size = _get_tensor_memory()

    print(size, size0)
    if size0 != size:
        ntries = 10
        _show_memsize(fcn, ntries, gccollect=gccollect)

    return size0 != size
