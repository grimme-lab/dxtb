"""
Reparametrisation of GFN-xTB
---------------------

Module for reparametrisation of GFN-xTB parametrisation. The fit of new parameters
is based on gradient descent, and automatically conducted using the pytorch framework.
Hence, new parameters are only dependent on train data and training setup.

Example
-------
>>> from pathlib import Path
>>> from dxtb.data.dataset import SampleDataset
>>> from dxtb.optimizer import reparametrise
>>> from dxtb.utils.utils import get_all_entries_from_dict
>>> # load data as batched sample
>>> path = "dxtb" / "data" / "PTB"
>>> dataset = SampleDataset.from_json(sorted(path.glob("samples_HCNO.json")))
>>> # only neutral samples
>>> dataset.samples = [s for s in dataset.samples if s.charges == torch.tensor(0)]
>>> # only small samples
>>> dataset.samples = [s for s in dataset.samples if s.positions.shape[0] <= 50] 
>>> # select all parameters
>>> names = get_all_entries_from_dict(GFN1_XTB, "hamiltonian.xtb.kpair")
>>> # select specific parameters
>>> # names = ["hamiltonian.kcn", "hamiltonian.hscale", "hamiltonian.shpoly", "hamiltonian.xtb.kpair['H-H']"]
>>> # NOTE: Add attributes directly to Param object. The Calculator constructor handles the correct
>>> #       assignment of unique elements and their orbitals and shells in batch 
>>> reparametrise(dataset, parameters=names, epochs=20)

"""

from __future__ import annotations
from typing import Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from ..typing import Tensor
from ..data.dataset import SampleDataset
from ..data.samples import Sample
from ..xtb.calculator import Calculator
from ..param.gfn1 import GFN1_XTB
from ..param.base import Param


class ParameterOptimizer(nn.Module):
    """Pytorch model for gradient optimization of given forward() function."""

    def __init__(self, parametrisation: Param, names: list[str]):
        super().__init__()
        assert id(parametrisation) != id(
            GFN1_XTB
        ), "Don't mess with global variables. Please use a copy of the GFN-xTB parametrisation."

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

    def forward(self, x: Sample):
        """
        Calculate singlepoint to obtain total energy and force gradient.
        """

        # detached graph to reduce the RAM footprint
        positions = x.positions.detach().clone()
        positions = positions.double()
        positions.requires_grad_(True)

        numbers = x.numbers.detach().clone()
        charges = x.charges.detach().clone()

        # calculate singlepoint
        calc = Calculator(numbers, positions, self.parametrisation)
        results = calc.singlepoint(numbers, positions, charges, verbosity=0)

        # total energy including repulsion, dispersion and halogen contribution
        energy = results.total

        # calculate gradients
        gradient = self.calc_gradient(energy, positions)

        return energy, gradient

    def calc_gradient(self, energy: Tensor, positions: Tensor) -> Tensor:
        """Calculate force gradient via autograd.

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


def train(
    model: ParameterOptimizer,
    dataloader: DataLoader,
    optimizer: torch.optim,
    loss_fn: Callable,
    epochs: int = 100,
    verbose: bool = False,
) -> list[Tensor]:
    """Optmisation loop conducting gradient descent.

    Args:
        model (ParameterOptimizer): Model containing trainable parametrisation.
        dataloader (DataLoader): Dataloader containing training data.
        optimizer (torch.optim): Optimizer for updating parameters.
        loss_fn (Callable): Criterion for calculation of loss.
        epochs (int, optional): Number of epochs. Defaults to 100.
        verbose (bool, optional): Additional print output. Defaults to False.

    Returns:
        list[Tensor]: Epochwise losses
    """

    losses = []

    for i in range(epochs):
        if verbose:
            print(f"Epoch {i}")
        for batch in dataloader:
            # conduct train step

            optimizer.zero_grad(set_to_none=True)

            # compare against reference gradient
            energy, grad = model(batch)
            loss = loss_fn(grad, batch.gref.double())
            # alternative: Jacobian of loss

            # calculate gradient and update model parameters
            loss.backward(inputs=model.params)
            optimizer.step()

            losses.append(loss.detach().item())
            if verbose:
                print(f"Batch: {batch}")
                print(f"Loss: {loss}")

    return losses


def reparametrise(
    dataset: SampleDataset,
    parameters: list[str],
    outpath: str | Path | None = None,
    **kwargs,
) -> list[Tensor]:

    # dynamic packing for each batch
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=Sample.pack,
        drop_last=False,
    )

    # GFN-xTB parametrisation to be optimised
    parametrisation = kwargs.get("parametrisation", GFN1_XTB.copy(deep=True))

    # setup model, optimizer and loss function
    model = ParameterOptimizer(parametrisation, parameters)
    lr = kwargs.get("lr", 0.1)
    opt = kwargs.get("optimizer", torch.optim.Adam(model.parameters(), lr=lr))
    loss_fn = kwargs.get("loss_fn", torch.nn.MSELoss(reduction="sum"))
    epochs = kwargs.get("epochs", 100)

    # print("Initial model.params", model.params)
    # for name, param in zip(parameters, model.params):
    #     print(name, param)

    # train model
    losses = train(model, dataloader, opt, loss_fn, epochs)

    # print("Final model.params", model.params)
    # for name, param in zip(parameters, model.params):
    #     print(name, param)

    # save new parametrisation to file
    if outpath is not None:
        # convert 0-d tensors to floats
        for name in parameters:
            v = model.parametrisation.get_param(name)
            if v.shape == torch.Size([]):
                model.parametrisation.set_param(name, v.item())

        model.parametrisation.to_toml(outpath, overwrite=True)

    # NOTE: also inference possible via '''predicted_energy, predicted_gradient = model(batch)'''
    return losses
