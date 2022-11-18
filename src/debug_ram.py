import torch
from torch import nn
from pathlib import Path

from memory_profiler import profile
import gc
import sys
import psutil
import os

# from dxtb.data.samples import Sample
from dxtb.data.dataset import SampleDataset
from dxtb.xtb.calculator import Calculator
from dxtb.param.gfn1 import GFN1_XTB
from dxtb.param.base import Param
from dxtb.utils.utils import get_all_entries_from_dict
from dxtb.scf.iterator import cpuStats

"""Script for investigating RAM usage"""


def zero_grad_(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()


def memReport():
    a = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            # print(type(obj), obj.size())
            a.append(obj)
    print(f"memReport: {len(a)} tensor objects")


def print_alive_tensors():
    # prints currently alive Tensors and Variables
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                print(type(obj), obj.size())
        except Exception as e:
            print(e)
            pass


def get_leaking_objects(objects=None):
    """Return objects that do not have any referents.

    These could indicate reference-counting bugs in C code.  Or they could
    be legitimate.

    Note that the GC does not track simple objects like int or str.

    .. versionadded:: 1.7
    """
    if objects is None:
        gc.collect()
        objects = gc.get_objects()
    try:
        ids = set(id(i) for i in objects)
        for i in objects:
            ids.difference_update(id(j) for j in gc.get_referents(i))
        # this then is our set of objects without referrers
        return [i for i in objects if id(i) in ids]
    finally:
        objects = i = j = None  # clear cyclic references to frame
    # NOTE: already from import standard libraries >100 entries


def count(typename, objects=None):
    """Count objects tracked by the garbage collector with a given class name.

    Example:

        >>> count('dict')
        42
        >>> count('MyClass', get_leaking_objects())
        3

    Note that the GC does not track simple objects like int or str.

    .. versionchanged:: 1.7
       New parameter: ``objects``.

    """
    if objects is None:
        objects = gc.get_objects()
    return sum(1 for o in objects if type(o).__name__ == typename)


def typestats(objects=None):
    """Count the number of instances for each type tracked by the GC.

    Note that the GC does not track simple objects like int or str.

    Note that classes with the same name but defined in different modules
    will be lumped together.

    Example:

        >>> typestats()
        {'list': 12041, 'tuple': 10245, ...}
        >>> typestats(get_leaking_objects())
        {'MemoryError': 1, 'tuple': 2795, 'RuntimeError': 1, 'list': 47, ...}

    .. versionadded:: 1.1

    .. versionchanged:: 1.7
       New parameter: ``objects``.

    """
    if objects is None:
        objects = gc.get_objects()
    stats = {}
    for o in objects:
        stats.setdefault(type(o).__name__, 0)
        stats[type(o).__name__] += 1
    return stats


def calc_gradient(energy, positions):

    print("doing the grad")

    gradient = torch.autograd.grad(
        energy.sum(),
        positions,
        create_graph=True,
        # also works: https://discuss.pytorch.org/t/using-the-gradients-in-a-loss-function/10821
        # grad_outputs=energy.sum().data.new(energy.sum().shape).fill_(1),
    )[0]

    print("finished the grad")
    return gradient


def forward(sample):

    # sample.positions.requires_grad_(False)
    sample.positions.requires_grad_(True)

    # TODO: maybe clone(), .data() or deepcopy those?
    positions = sample.positions.detach().clone()
    numbers = sample.numbers.detach().clone()
    charges = sample.charges.detach().clone()
    positions.requires_grad_(True)

    assert id(sample.positions) != id(positions)
    assert id(GFN1_XTB.copy(deep=True)) != id(GFN1_XTB)

    calc = Calculator(numbers, positions, GFN1_XTB.copy(deep=True))
    # energy = calc.dummypoint(numbers, positions, charges, verbosity=0)
    results = calc.singlepoint(numbers, positions, charges, verbosity=0)
    energy = results.total
    print("AFTER SINGLEPOINT")
    cpuStats()

    # TODO: if sample.positions is taken, all is ok! (or at least drops significantly faster!)
    #   --> as soon as positions require grad, the RAM grows (i.e. the gradients are not properly removed)

    # TODO: es gibt zwei baustellen:
    #   1. static RAM overflow: singlepoint increases RAM even when pos.no_grad
    #   2. dynamic RAM overflow: singlepoint increases RAM when pos.grad
    #   3. how does grad calculation affect RAM
    #   4. how does loss calculation of grad affect RAM

    # TODO: to avoid RAM overflow maybe zero_grad parameters withhin model.parametrisation (and model.params)
    # TODO: gradients (forces) should also be freed(?)
    # TODO: for small(why?) systems the increment of SCF is actually big

    gradient = calc_gradient(energy, positions)
    # TODO: this creates a new SCF._data object
    # gradient = None
    return energy, gradient


def main():
    print("hello")

    cpuStats()
    memReport()

    # load data
    path = Path(__file__).resolve().parents[1] / "data" / "PTB" / "samples_HCNO.json"
    dataset = SampleDataset.from_json(path)

    # dataset.samples = [s for s in dataset.samples if s.positions.shape[0] >= 100]

    cpuStats()
    memReport()

    # names = ["hamiltonian.xtb.kpair['H-H']"]
    # params = [
    #     torch.tensor(GFN1_XTB.get_param(name), dtype=torch.float64, requires_grad=True)
    #     for name in names
    # ]
    # optimizer = torch.optim.SGD(params, lr=0.1)
    # loss_fn = torch.nn.MSELoss(reduction="sum")

    import copy

    def calc(batch):

        energy, grad = forward(batch)

        # zero_grad_(batch.positions)
        # batch.positions.requires_grad_(False)

        # print(batch.positions.grad)
        del batch
        del energy, grad

    for i in range(100):
        print(f"epoch {i}")
        for j in range(len(dataset)):
            # batch = dataset[j]
            # batch = copy.deepcopy(dataset[j])
            # assert id(batch) != id(dataset[j])
            # print(f"batch: {j} | {batch}")

            # optimizer.zero_grad(set_to_none=True)

            # print(batch.positions.grad)

            calc(copy.deepcopy(dataset[j]))

            print(("cuda_allocated: %d Mb", torch.cuda.memory_allocated() // (1 << 20)))
            torch.cuda.empty_cache()
            gc.collect()
            # I.
            # 1. pypy GC collector?
            # 2. xitorch leak
            # 3. overlap leak
            # 4. cPy GC more aggressive

            cpuStats()
            memReport()

            print("here we are")
            # print_alive_tensors()

            # already here growing number of tensor objects (and hence RAM)
            """energy, grad = model(batch)
            y_true = batch.gref.double()
            loss = loss_fn(grad, y_true)
            # alternative: Jacobian of loss

            # calculate gradient to update model parameters
            loss.backward(inputs=model.params)

            optimizer.step()"""

    # number of SCF instances
    print("SCF count", count("SelfConsistentField"))
    # TODO: this is the RAM problem (should be == 0!)
    # this equals the number of epochs * number_samples

    print("ALmost last mem report")
    cpuStats()
    memReport()

    print("Finished")
    return

    """import ctypes
    aaa = ctypes.cast(id(abc), ctypes.py_object).value
    del aaa"""

    def print_refcounts():
        for i in SelfConsistentField.instances:
            # print(type(i), id(i))
            print(f"Refcnt {i}", sys.getrefcount(i))

    print("SCF count", count("SelfConsistentField"))
    print("DS count", count("SampleDataset"))
    print("GFN1_XTB count", count("GFN1_XTB"))

    print([sys.getrefcount(i) for i in SelfConsistentField.instances])

    print_refcounts()

    # delete SCF
    SelfConsistentField.instances = []
    SampleDataset.samples = []
    dataset.samples = []
    del dataset
    assert count("SelfConsistentField") == 0
    assert count("Calculator") == 0
    assert count("SampleDataset") == 0

    del SelfConsistentField

    gc.collect()

    # TODO: what is still alive at all?!
    print(locals())
    # print(globals())
    print(len(globals()))
    # del SampleDataset
    global calc_gradient
    del calc_gradient
    global GFN1_XTB
    del GFN1_XTB
    print(globals())
    print(len(globals()))

    print(dir())

    gc.collect()

    print("Final mem report")
    cpuStats()
    memReport()


if __name__ == "__main__":
    main()
