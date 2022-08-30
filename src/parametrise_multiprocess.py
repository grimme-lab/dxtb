from __future__ import annotations
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from multiprocessing import Process

import gc
import os.path as op
import tomli as toml

from xtbml.data.dataset import SampleDataset
from xtbml.data.samples import Sample
from xtbml.xtb.calculator import Calculator
from xtbml.param.gfn1 import GFN1_XTB
from xtbml.param.base import Param
from xtbml.utils.utils import get_all_entries_from_dict
from xtbml.typing import Tensor

"""Script for investigating RAM usage"""
from xtbml.scf.iterator import cpuStats


EPOCHS = 100
LEARNING_RATE = 0.01
DATA_SET = "Amino20x4"


"""
ACONF: MD= -0.660 MAD=  0.660 RMS= 0.77472576
SCONF: MD= -0.912 MAD=  2.502 RMS= 4.62769921
PCONF: MD=  0.859 MAD=  2.171 RMS= 2.73447842
Amino: MD= -0.546 MAD=  1.114 RMS= 1.38088377
MCONF: MD= -1.377 MAD=  1.443 RMS= 1.63631580
"""


def zero_grad_(x):
    if x.grad is not None:
        x.grad.detach_()
        x.grad.zero_()


def memReport():
    a = [o for o in gc.get_objects() if torch.is_tensor(o)]
    print(f"memReport: {len(a)} tensor objects")


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

    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        positions = x.positions.detach().clone().double()
        positions.requires_grad_(True)
        numbers = x.numbers.detach().clone()
        charges = x.charges.detach().clone()

        def get_energy(numbers, positions, parametrisation):
            calc = Calculator(numbers, positions, parametrisation)
            results = calc.singlepoint(numbers, positions, charges, verbosity=0)
            energy = results.total
            gc.collect()  # garbage collection
            # TODO: maybe some part of the calculator object remains and still holds some values?
            return energy.double()

        energy = get_energy(numbers, positions, self.parametrisation)
        gradient = self.calc_gradient(energy, positions)

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
            # grad_outputs=energy.sum().data.new(energy.sum().shape).fill_(1),
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

    return loss


def training_epoch(dataloader, names, epoch):

    losses = []

    # reload GFN1_XTB parametrisation in every epoch
    root = Path(__file__).resolve().parents[0]
    file = "gfn1-xtb_ORIGINAL.toml" if epoch == 0 else "gfn1-xtb_tmp.toml"
    gfn1_xtb = load_parametrisation(root / file)

    # setup model, optimizer and loss function
    model = ParameterOptimizer(gfn1_xtb, names)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    if epoch != 0:
        # reload optimizer and model from disk
        losses = load_checkpoint(root / f"gfn1-xtb_tmp.pt", model, optimizer)

    print("INITAL model.params", model.params)
    for name, param in zip(names, model.params):
        print(name, param)

    for batch in dataloader:
        print(f"batch: {batch}")

        loss = train_step(optimizer, model, batch, loss_fn)
        print(loss)
        losses.append(loss.item())

        gc.collect()  # garbage collection
        # TODO: check https://stackify.com/python-garbage-collection/ for detailed gc

    # TODO: check which part of comp graph is overflowing here

    # convert 0-d tensors to floats
    for name in names:
        v = model.parametrisation.get_param(name)
        if v.shape == torch.Size([]):
            model.parametrisation.set_param(name, v.item())

    print("saving parametrisation")
    save_parametrisation(gfn1_xtb, root / f"gfn1-xtb_tmp.toml")
    save_checkpoint(model, optimizer, path=root / f"gfn1-xtb_tmp.pt", losses=losses)

    print("FINAL model.params", model.params)
    for name, param in zip(names, model.params):
        print(name, param)

    cpuStats()
    memReport()

    print(f"END of EPOCH {epoch}")
    print("losses ", losses)


def load_parametrisation(path=op.join(op.dirname(__file__), "gfn1-xtb.toml")):
    with open(path, "rb") as fd:
        gfn1_xtb = Param(**toml.load(fd))
    return gfn1_xtb


def save_parametrisation(param, path=op.join(op.dirname(__file__), "gfn1-xtb.toml")):
    param.to_toml(path=path, overwrite=True)


def load_checkpoint(path, model, optimizer, losses=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["losses"]


def save_checkpoint(model, optimizer, path, losses=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
        },
        path,
    )


def parametrise_dyn_loading():

    cpuStats()
    memReport()

    # load data as batched sample
    path1 = Path(__file__).resolve().parents[1] / "data" / "ACONF"
    path2 = Path(__file__).resolve().parents[1] / "data" / "MCONF"
    path3 = Path(__file__).resolve().parents[1] / "data" / "SCONF"
    path4 = Path(__file__).resolve().parents[1] / "data" / "PCONF21"
    path5 = Path(__file__).resolve().parents[1] / "data" / "Amino20x4"

    paths = [path1 / "samples.json"]
    paths = [
        path1 / "samples.json",
        path2 / "samples.json",
        path3 / "samples.json",
        path4 / "samples.json",
        path5 / "samples.json",
    ]

    dataset = SampleDataset.from_json(paths)
    cpuStats()
    memReport()

    # dataset.samples = [s for s in dataset.samples if s.positions.shape[0] >= 100]
    dataset.samples = [dataset.samples[0]]

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

    names = get_all_entries_from_dict(GFN1_XTB, "hamiltonian.xtb.kpair")

    names = names + [
        "hamiltonian.xtb.enscale",
        "hamiltonian.xtb.kpol",
        "hamiltonian.xtb.shell['ss']",
        "hamiltonian.xtb.shell['pp']",
        "hamiltonian.xtb.shell['dd']",
        "hamiltonian.xtb.shell['sp']",
        "repulsion.effective.kexp",
        "dispersion.d3.s6",
        "dispersion.d3.s8",
        "dispersion.d3.a1",
        "dispersion.d3.a2",
        # "charge.effective.gexp", # NaN error
        # "halogen.classical.damping",
        # "halogen.classical.rscale",
        # "hamiltonian.xtb.kpair['H-H']",
        # "hamiltonian.xtb.kpair['B-H']",
        # "hamiltonian.xtb.kpair['N-H']",
        # "hamiltonian.xtb.kpair['Si-N']",
    ]

    # cpuStats()
    # memReport()

    for i in range(EPOCHS):
        print(f"epoch {i}")

        # spawn subprocess for each epoch (avoid memory leak)
        p = Process(target=training_epoch, args=(dataloader, names, i))
        p.start()
        p.join()  # this blocks until the process terminates

    print("finished")
    return


if __name__ == "__main__":
    parametrise_dyn_loading()
