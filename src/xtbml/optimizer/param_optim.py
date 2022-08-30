from __future__ import annotations
from typing import Callable, List
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from memory_profiler import profile
import gc
import sys
import psutil
import os

from ..typing import Tensor
from ..data.dataset import SampleDataset
from ..data.samples import Sample
from ..xtb.calculator import Calculator
from ..param.gfn1 import GFN1_XTB
from ..param.base import Param
from ..constants import AU2KCAL
from ..utils.utils import get_all_entries_from_dict


def zero_grad_(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()


def memReport():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            print(type(obj), obj.size())


def cpuStats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())  # physical memory usage
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory GB:", memoryUse)


class ParameterOptimizer(nn.Module):
    """Pytorch model for gradient optimization of given forward() function."""

    def __init__(self, parametrisation: Param, names: list[str]):
        super().__init__()

        # don't mess with global variables
        assert id(parametrisation) != id(GFN1_XTB)

        # parametrisation as proprietary attribute
        self.parametrisation = parametrisation

        # get parameters from parametrisation
        params = [
            torch.tensor(
                self.parametrisation.get_param(name),
                dtype=torch.float64,
            )
            for name in names
        ]

        # register as learnable model parameters
        self.params = nn.ParameterList([nn.Parameter(p) for p in params])

        # assign parameters to parametrisation
        for name, param in zip(names, self.params):
            self.parametrisation.set_param(name, param)

        # TODO: check that params are updated in self.parametrisation during multiple batches
        # TODO: check that params are not overwritten in Calculator setup (i.e. floats converted to tensors)

    @profile
    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        if False:

            positions = x.positions.detach()
            positions = positions.double()
            positions.requires_grad_(True)

            # detached graph for reduce the RAM footprint
            numbers = x.numbers.detach()
            charges = x.charges.detach()

            calc = Calculator(numbers, positions, self.parametrisation)
            # TODO: need to free all other parameters in self.parametrisation from graph?
            #       --> should already be only floats (except for model params?)

            # calculate energies
            results = calc.singlepoint(numbers, positions, charges, verbosity=0)

            # total energy including repulsion, dispersion and halogen contribution
            energy = results.total

            # calculate gradients
            gradient = self.calc_gradient(energy, positions)

            return energy, gradient

        if False:
            # temporary testing
            dim = x.numbers.shape

            energy = torch.ones_like(x.numbers, dtype=torch.float64).requires_grad_(
                True
            )
            gradient = torch.ones(
                [dim[0], dim[1], 3], dtype=torch.float64
            ).requires_grad_(True)
            # TODO: even this way, 0.5Mb RAM increase per epoch

        #####################

        # TODO: maybe clone(), .data() or deepcopy those?
        positions = x.positions.detach().clone().double()
        positions.requires_grad_(True)
        numbers = x.numbers.detach().clone()
        charges = x.charges.detach().clone()

        assert id(x.positions) != id(positions)
        assert id(x.charges) != id(charges)
        assert id(x.numbers) != id(numbers)

        def get_energy(numbers, positions, parametrisation):

            calc = Calculator(numbers, positions, parametrisation)
            # return (
            #     torch.sum(positions) * parametrisation.hamiltonian.xtb.kpair["H-H"]
            # )  # TODO: that solves the RAM issue

            # TODO: somewhere inside singlepoint - sth is not freed correctly
            # results = calc.dummypoint(numbers, positions, charges, verbosity=0)
            results = calc.singlepoint(numbers, positions, charges, verbosity=0)
            # TODO: raises RAM footprint

            # TODO: overlap and SCF cause the largest RAM increase

            del calc
            gc.collect()  # garbage collection
            # TODO: maybe some part of the calculator object remains and still holds some values?

            return results.total.double()

        print("compare")
        energy = get_energy(numbers, positions, self.parametrisation)
        print(energy)
        # del results
        print("here we are")

        gradient = self.calc_gradient(energy, positions)
        # TODO: raises RAM footprint
        # gradient = None

        # TODO: for total HCNO it seems that gradient is main reason for rising RAM
        #       -- how to delete gradient and free computational graph?

        #####################

        del positions
        del numbers
        del charges

        gc.collect()  # garbage collection

        return energy, gradient

    def calc_gradient(self, energy: Tensor, positions: Tensor) -> Tensor:
        """Calculate gradient (i.e. force) via pytorch autodiff framework.

        Parameters
        ----------
        energy : Tensor
            Energy output for given prediction model
        positions : Tensor
            Positions for given system

        Returns
        -------
        Tensor
            Force tensor dE/dxyz
        """
        gradient = torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=True,
            grad_outputs=energy.sum().data.new(energy.sum().shape).fill_(1),
            # also works: https://discuss.pytorch.org/t/using-the-gradients-in-a-loss-function/10821
        )[0]
        return gradient


def train_step(optimizer, model, batch, loss_fn):

    optimizer.zero_grad(set_to_none=True)

    energy, grad = model(batch)
    y_true = batch.gref.double()
    loss = loss_fn(grad, y_true)
    # alternative: Jacobian of loss

    # calculate gradient to update model parameters
    loss.backward(inputs=model.params)

    optimizer.step()

    """print(grad)
    for t in [grad, loss]:
        t.detach_()
        t.zero_()
        # t.grad.detach_()
        # t.grad.zero_()
    model.zero_grad(set_to_none=True)"""

    del energy
    del grad
    del y_true

    return loss


def training_loop(
    model: nn.Module,
    optimizer: torch.optim,
    loss_fn: Callable,
    dataloader: DataLoader,
    n: int = 100,
    verbose: bool = True,
) -> List[Tensor]:
    "Optmisation loop conducting gradient descent."
    losses = []

    for i in range(n):
        if verbose:
            print(f"epoch {i}")
        for batch in dataloader:
            if verbose:
                print(f"batch: {batch}")

            '''optimizer.zero_grad(set_to_none=True)

            """print("check gradients BEFORE")
            print(model.params[0].grad)
            print(model.parametrisation.hamiltonian.xtb.kpair["H-H"].grad)
            # print("Memory allocated", torch.cuda.memory_allocated())

            print("-----------")"""

            energy, grad = model(batch)
            y_true = batch.gref.double()
            loss = loss_fn(grad, y_true)
            # alternative: Jacobian of loss

            # calculate gradient to update model parameters
            loss.backward(inputs=model.params)

            optimizer.step()'''

            loss = train_step(optimizer, model, batch, loss_fn)
            # losses.append(loss.item())
            losses.append(loss.detach().item())
            if verbose:
                print(loss)

            # TODO: to avoid RAM overflow maybe zero_grad parameters withhin model.parametrisation (and model.params)
            # TODO: gradients (forces) should also be freed(?)
            # TODO: for small(why?) systems the increment of SCF is actually big

            # del energy
            # del grad
            del loss

            gc.collect()  # garbage collection
            print("GARBAGE", gc.garbage)

            """print("check gradients AFTER")
            print(model.params[0].grad)
            print(model.parametrisation.hamiltonian.xtb.kpair["H-H"].grad)
            print(id(model.params[0].grad))
            print(id(model.parametrisation.hamiltonian.xtb.kpair["H-H"].grad))

            def zero_grad(tensor):
                tensor.grad.detach_()
                tensor.grad.zero_()

            # print("Memory allocated", torch.cuda.memory_allocated())
            print("-----------")"""

    """print("check what part of comp graph is overflowing here")
    import sys

    # sys.exit(0)"""

    return losses


def plot_losses(losses: list[float], path: str = "losses.png"):

    import matplotlib.pyplot as plt
    import numpy as np

    plt.plot(np.array(losses), "r")
    plt.savefig(path)


def example():
    @profile
    def get_ptb_dataset(root: Path) -> SampleDataset:
        list_of_path = sorted(
            root.glob("samples_DUMMY.json")
        )  # samples_*.json samples_HE*.json samples_DUMMY.json samples_HCNO_debug
        return SampleDataset.from_json(list_of_path)

    # load data as batched sample
    path = Path(__file__).resolve().parents[3] / "data" / "PTB"
    dataset = get_ptb_dataset(path)

    # only neutral samples
    # dataset.samples = [s for s in dataset.samples if s.charges == torch.tensor(0)]
    # small samples only
    # dataset.samples = [s for s in dataset.samples if s.positions.shape[0] <= 50]
    # dataset.samples = [s for s in dataset.samples if s.positions.shape[0] >= 50]

    # batch = Sample.pack(dataset.samples)
    print("Number samples", len(dataset))

    # dynamic packing for each batch
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Sample.pack,
        drop_last=False,
    )

    # NOTE: Add attributes directly to Param object. The Calculator constructor handles the correct
    #       assignment of unique elements and their orbitals and shells in batch

    names = get_all_entries_from_dict(GFN1_XTB, "hamiltonian.xtb.kpair")
    # names = ["hamiltonian.kcn", "hamiltonian.hscale", "hamiltonian.shpoly"]
    # names = [
    #     "hamiltonian.xtb.kpair['H-H']",
    #     "hamiltonian.xtb.kpair['B-H']",
    #     "hamiltonian.xtb.kpair['N-H']",
    #     "hamiltonian.xtb.kpair['Si-N']",
    # ]
    names = ["hamiltonian.xtb.kpair['H-H']"]

    # TODO: still a lot of RAM groth during batches/epochs (are SCF calculation entirely freed from RAM?)
    # TODO: check garbage collector

    # TODO: write test, that only H-H and H-O changes for water

    """tensor(0.0224, dtype=torch.float64, grad_fn=<MseLossBackward0>)
    GARBAGE []
    FINAL model.params ParameterList(  (0): Parameter containing: [torch.DoubleTensor of size ])
    hamiltonian.xtb.kpair['H-H'] Parameter containing:
    tensor(1.3564, dtype=torch.float64, requires_grad=True)"""

    # setup model, optimizer and loss function
    model = ParameterOptimizer(GFN1_XTB.copy(deep=True), names)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    # optmizier options: Adagrad, RMSPROP

    print("INITAL model.params", model.params)
    for name, param in zip(names, model.params):
        print(name, param)

    losses = training_loop(model, opt, loss_fn, dataloader, n=4)

    print("FINAL model.params", model.params)
    for name, param in zip(names, model.params):
        print(name, param)

    cpuStats()

    # convert 0-d tensors to floats
    for name in names:
        v = model.parametrisation.get_param(name)
        if v.shape == torch.Size([]):
            model.parametrisation.set_param(name, v.item())

    return

    # save new parametrisation
    model.parametrisation.to_toml("gfn1-xtb_out.toml", overwrite=True)

    print("Losses: ", losses)
    plot_losses(losses)
    return

    # inference
    preds, grad = model(batch)

    print("loss")
    lgfn1 = loss_fn(batch.ggfn1, batch.gref)
    lopt = loss_fn(grad, batch.gref)
    print(lgfn1)
    print(lopt)

    print("prediction")  # NOTE: energies in JSON in kcal/mol
    print(batch.egfn1)
    print(preds * AU2KCAL)
    print(batch.eref)

    # print("grad comparison")
    # print(batch.ggfn1)
    # print(grad)
    # print(batch.gref)

    return


# for single vs einmal über total data
# names = all_toml
# positions.no_grad
# e = singlepoint
# e.backward()
# print(model.params.grad)
