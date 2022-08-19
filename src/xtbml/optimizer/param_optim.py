from __future__ import annotations
from typing import Callable, List
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from tad_dftd3 import dftd3

from ..typing import Tensor
from ..data.dataset import SampleDataset
from ..data.samples import Sample
from ..xtb.calculator import Calculator
from ..param.gfn1 import GFN1_XTB
from ..constants import AU2KCAL


class ParameterOptimizer(nn.Module):
    """Pytorch model for gradient optimization of given forward() function."""

    def __init__(self, params: list[Tensor], params_name: list[str]):
        super().__init__()

        # register as learnable model parameters
        self.params = nn.ParameterList([nn.Parameter(param) for param in params])

        # name of parameter within calculator
        self.params_name = params_name

    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        positions = x.positions.detach()
        positions = positions.double()
        positions.requires_grad_(True)

        calc = Calculator(x.numbers, positions, GFN1_XTB)

        # assign model weights (= learnable parameter) to calculator
        for i, param in enumerate(self.params):
            Calculator.set_param(calc, self.params_name[i], self.params[i])

        # calculate energies
        results = calc.singlepoint(x.numbers, positions, x.charges, verbosity=0)
        edisp = self.calc_dispersion(x.numbers, positions)

        energy = results["energy"] + edisp

        # calculate gradients
        gradient = self.calc_gradient(energy, positions)

        return energy, gradient

    def calc_dispersion(self, numbers: Tensor, positions: Tensor) -> Tensor:
        """Calculate D3 dispersion correction for given sample.

        Args:
            numbers (Tensor): Atomic charges for sample
            positions (Tensor): Atomic positions for sample

        Returns:
            Tensor: The dispersion correction edisp
        """
        pdisp = {
            "s6": GFN1_XTB.dispersion.d3.s6,
            "s8": GFN1_XTB.dispersion.d3.s8,
            "a1": GFN1_XTB.dispersion.d3.a1,
            "a2": GFN1_XTB.dispersion.d3.a2,
        }
        return dftd3(numbers, positions, pdisp)

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
        )[0]
        return gradient


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
        for batch in dataloader:
            if verbose:
                print(f"epoch {i}")
            optimizer.zero_grad()
            energy, grad = model(batch)
            y_true = batch.gref.double()
            loss = loss_fn(grad, y_true)
            # alternative: Jacobian of loss

            # calculate gradient to update model parameters
            loss.backward(inputs=model.params)

            optimizer.step()
            losses.append(loss)
            if verbose:
                print(loss)
    return losses


def example():
    def get_ptb_dataset(root: Path) -> SampleDataset:
        list_of_path = sorted(
            root.glob("samples_HCNO.json")
        )  # samples_*.json samples_HE*.json samples_DUMMY.json
        return SampleDataset.from_json(list_of_path)

    # load data as batched sample
    path = Path(__file__).resolve().parents[3] / "data" / "PTB"
    dataset = get_ptb_dataset(path)
    batch = Sample.pack(dataset.samples)
    print(type(batch))
    print(batch.numbers.shape)

    # dynamic packing for each batch
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=Sample.pack,
        drop_last=False,
    )

    # setup calculator and get parameters
    calc = Calculator(batch.numbers, batch.positions, GFN1_XTB)
    names = ["hamiltonian.kcn", "hamiltonian.hscale", "hamiltonian.shpoly"]
    params_to_opt = [Calculator.get_param(calc, name) for name in names]
    # TODO: check dataloader is setup for _all_ elements in samples, during batch only limited might occur
    #       --> does that work with calculator internal unique element representation in batch?

    # setup model, optimizer and loss function
    model = ParameterOptimizer(params_to_opt, names)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    print("INITAL model.params", model.params)
    for param in model.params:
        print(param)

    losses = training_loop(model, opt, loss_fn, dataloader, n=100)

    print("FINAL model.params", model.params)
    for param in model.params:
        print(param)

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
