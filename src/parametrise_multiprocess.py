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


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.eps = eps

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y) + self.eps)


EPOCHS = 100
LEARNING_RATE = 0.01
LOSS_FN = RMSELoss()
# LOSS_FN = torch.nn.MSELoss(reduction="sum")
FILE = f"mconf_{EPOCHS}_{LEARNING_RATE}"

torch.autograd.set_detect_anomaly(True)

"""
ACONF: MD= -0.660 MAD=  0.660 RMS= 0.77472576
SCONF: MD= -0.912 MAD=  2.502 RMS= 4.62769921
PCONF: MD=  0.859 MAD=  2.171 RMS= 2.73447842
Amino: MD= -0.546 MAD=  1.114 RMS= 1.38088377
MCONF: MD= -1.377 MAD=  1.443 RMS= 1.63631580
ADIM6: MD= -1.007 MAD=  1.007 RMS= 1.07390254
BUT14: MD= -0.736 MAD=  0.953 RMS= 1.13553333
IDISP: MD= -6.208 MAE=  6.528 RMS= 10.37706686
UPU23: MD=  0.028 MAE=  1.239 RMS= 1.44304479
S22:   MD= -1.275 MAE=  1.330 RMS= 1.66241746
S66:   MD= -1.052 MAE=  1.081 RMS= 1.23126813
ICONF: MD= -2.533 MAE=  2.627 RMS= 4.07162202
WATER: MD= -0.001 MAE=  7.514 RMS= 9.25648321

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

        self.names = names

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


def l1_regularisation(model: ParameterOptimizer, parametrisation: Param):

    reg_loss = 0
    for name, p in zip(model.names, model.parameters()):
        p_org = parametrisation.get_param(name)
        reg_loss += torch.abs((torch.abs(p - p_org)) / p_org)

    # TODO: think about turning off gradients here
    return reg_loss


def train_step(optimizer, model, batch, loss_fn):

    optimizer.zero_grad(set_to_none=True)

    energy, grad = model(batch)
    y_true = batch.gref.double()
    loss = loss_fn(grad, y_true)
    # alternative: Jacobian of loss

    # optional: L1 regularisation
    #           NOTE: parameter difference to initial parameters is used as penalty
    l1_lambda = 0.0005

    reg = l1_lambda * l1_regularisation(model, GFN1_XTB)
    print("loss =", loss.item(), " ; reg =", reg.item())
    loss = loss + reg

    # calculate gradient to update model parameters
    loss.backward(inputs=model.params)

    optimizer.step()

    return loss


def training_epoch(dataloader, names, epoch):

    losses = []
    tracked_losses = []

    # reload GFN1_XTB parametrisation in every epoch
    root = Path(__file__).resolve().parents[0]
    file = "gfn1-xtb_ORIGINAL.toml" if epoch == 0 else f"{FILE}.toml"
    gfn1_xtb = load_parametrisation(root / file)

    # setup model, optimizer and loss function
    model = ParameterOptimizer(gfn1_xtb, names)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )  # NOTE: use weight decay as proxy for L2-normalisation

    if epoch != 0:
        # reload optimizer and model from disk
        losses, tracked_losses = load_checkpoint(root / f"{FILE}.pt", model, optimizer)

    print("INITAL model.params")
    for name, param in zip(names, model.params):
        print(name, param)
    print("")

    for batch in dataloader:
        print(f"batch: {batch}")

        loss = train_step(optimizer, model, batch, LOSS_FN)
        print(loss)
        losses.append(loss.item())

        gc.collect()  # garbage collection
        # TODO: check https://stackify.com/python-garbage-collection/ for detailed gc

    tracked_losses.append(sum(losses) / len(losses))

    # TODO: check which part of comp graph is overflowing here

    # convert 0-d tensors to floats
    for name in names:
        v = model.parametrisation.get_param(name)
        if v.shape == torch.Size([]):
            model.parametrisation.set_param(name, v.item())

    print("\nsaving parametrisation")
    save_parametrisation(gfn1_xtb, root / f"{FILE}.toml")
    save_checkpoint(
        model,
        optimizer,
        path=root / f"{FILE}.pt",
        losses=losses,
        tracked_losses=tracked_losses,
    )

    print("FINAL model.params")
    for name, param in zip(names, model.params):
        print(name, param)

    # cpuStats()
    # memReport()

    print(f"\nEND of EPOCH {epoch}")
    print("losses ", sum(losses) / len(losses))

    if epoch == EPOCHS - 1:
        # reload optimizer and model from disk
        _, tracked_losses = load_checkpoint(root / f"{FILE}.pt", model, optimizer)

        print("\nAll losses")
        [print(l) for l in tracked_losses]


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
    return checkpoint["losses"], checkpoint["tracked_losses"]


def save_checkpoint(model, optimizer, path, losses=None, tracked_losses=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "tracked_losses": tracked_losses,
        },
        path,
    )


def parametrise_dyn_loading():

    cpuStats()
    memReport()

    """
    ACONF:   ['H', 'C']
    MCONF:   ['H', 'C', 'N', 'O']
    SCONF:   ['H', 'C', 'O']
    PCONF21: ['H', 'C', 'N', 'O']
    Amino:   ['H', 'C', 'N', 'O', 'S']
    BUT14:   ['H', 'C', 'O']
    UPU23:   ['H', 'C', 'N', 'O', 'P']
    IDISP:   ['H', 'C']
    """

    # load data as batched sample
    path1 = Path(__file__).resolve().parents[1] / "data" / "ACONF" / "samples.json"
    path2 = Path(__file__).resolve().parents[1] / "data" / "MCONF" / "samples.json"
    path3 = Path(__file__).resolve().parents[1] / "data" / "SCONF" / "samples.json"
    path4 = Path(__file__).resolve().parents[1] / "data" / "PCONF21" / "samples.json"
    path5 = Path(__file__).resolve().parents[1] / "data" / "Amino20x4" / "samples.json"
    path6 = Path(__file__).resolve().parents[1] / "data" / "BUT14DIOL" / "samples.json"
    path7 = Path(__file__).resolve().parents[1] / "data" / "UPU23" / "samples.json"
    path8 = Path(__file__).resolve().parents[1] / "data" / "IDISP" / "samples.json"
    path9 = Path(__file__).resolve().parents[1] / "data" / "S22" / "samples.json"
    path10 = Path(__file__).resolve().parents[1] / "data" / "S66" / "samples.json"
    path11 = Path(__file__).resolve().parents[1] / "data" / "WATER27" / "samples.json"

    dataset = SampleDataset.from_json([path2])
    cpuStats()
    memReport()

    # dataset.samples = [s for s in dataset.samples if s.positions.shape[0] >= 100]
    # dataset.samples = [dataset.samples[0]]

    # batch = Sample.pack(dataset.samples)
    print("Number samples", len(dataset))
    print("Atom types", dataset.get_atom_types())

    # dynamic packing for each batch
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Sample.pack,
        drop_last=False,
    )

    # names = get_all_entries_from_dict(GFN1_XTB, "hamiltonian.xtb.kpair")

    names = [
        "hamiltonian.xtb.enscale",
        "hamiltonian.xtb.kpol",
        "hamiltonian.xtb.shell['ss']",
        "hamiltonian.xtb.shell['pp']",
        "hamiltonian.xtb.shell['dd']",
        "hamiltonian.xtb.shell['sp']",
        "repulsion.effective.kexp",
        # "dispersion.d3.s6",
        "dispersion.d3.s8",
        "dispersion.d3.a1",
        "dispersion.d3.a2",
        "charge.effective.gexp",
        "halogen.classical.damping",
        "halogen.classical.rscale",
        # "element['H'].arep",
        # "element['C'].arep",
        # "element['N'].arep",
        # "element['O'].arep",
        # "element['S'].arep",
        # "element['P'].arep",
        # "element['H'].zeff",
        # "element['C'].zeff",
        # "element['N'].zeff",
        # "element['O'].zeff",
        # "element['S'].zeff",
        # "element['P'].zeff",
        # "hamiltonian.xtb.kpair['H-H']",
        # "hamiltonian.xtb.kpair['B-H']",
        # "hamiltonian.xtb.kpair['N-H']",
        # "hamiltonian.xtb.kpair['Si-N']",
    ]

    # cpuStats()
    # memReport()

    for i in range(EPOCHS):
        print(f"\n\nepoch {i}")

        # spawn subprocess for each epoch (avoid memory leak)
        p = Process(target=training_epoch, args=(dataloader, names, i))
        p.start()
        p.join()  # this blocks until the process terminates

    print("finished")
    return


if __name__ == "__main__":
    parametrise_dyn_loading()
