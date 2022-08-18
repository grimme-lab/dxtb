from typing import Callable, List
import torch
from torch import nn
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

    def __init__(self, weights, weights_name):
        super().__init__()

        # register weights as model parameters
        self.weights = torch.nn.Parameter(weights)
        # TODO: allow for multiple parameters to be set
        # check via print(list(model.parameters()))

        # name of parameter within calculator
        self.weights_name = weights_name

    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        positions = x.positions.detach()
        positions = positions.double()
        positions.requires_grad_(True)

        calc = Calculator(x.numbers, positions, GFN1_XTB)

        # assign model weights (= learnable parameter) to calculator
        Calculator.set_param(calc, self.weights_name, self.weights)

        # calculate energies
        results = calc.singlepoint(x.numbers, positions, x.charges, verbosity=0)
        energy = results["energy"]
        # TODO: add dispersion correction

        # convert to kcal
        # energy = energy * AU2KCAL
        # TODO: why does this conversion change the training outcome of the gradient?
        #   could it be that the gradient in JSON is not converted to kcal? (i.e. |dE/dxyz|)

        # calculate gradients
        gradient = self.calc_gradient(energy, positions)

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
        )[0]
        return gradient


def training_loop(
    model: nn.Module, optimizer: torch.optim, loss_fn: Callable, x: Sample, n: int = 100
) -> List[Tensor]:
    "Optmisation loop conducting gradient descent."
    losses = []

    for i in range(n):
        print(f"epoch {i}")
        optimizer.zero_grad()
        energy, grad = model(x)
        y_true = x.gref.double()
        loss = loss_fn(grad, y_true)
        # alternative: Jacobian of loss

        # calculate gradient to update model parameters
        loss.backward(inputs=model.weights)

        optimizer.step()
        losses.append(loss)
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
    batch = Sample.pack(dataset.samples[5:6])
    # TODO: currently batch SCF not working (due to batching?)
    #       - check charge(?) batching with einsum mapping in SCF
    print(type(batch))
    print(batch.numbers.shape)

    # setup calculator and get parameters
    calc = Calculator(batch.numbers, batch.positions, GFN1_XTB)
    name = "hamiltonian.kcn"
    param_to_opt = Calculator.get_param(calc, name)

    # setup model, optimizer and loss function
    model = ParameterOptimizer(param_to_opt, name)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    print("INITAL model.weights", model.weights)

    losses = training_loop(model, opt, loss_fn, batch)

    print("FINAL model.weights", model.weights)

    # inference
    preds, grad = model(batch)

    print("loss")
    lgfn1 = loss_fn(batch.ggfn1, batch.gref)
    lopt = loss_fn(grad, batch.gref)
    print(lgfn1)
    print(lopt)

    print("prediction")
    print(batch.egfn1)
    print(preds * AU2KCAL)
    print(batch.eref)

    # print("grad comparison")
    # print(batch.ggfn1)
    # print(grad)
    # print(batch.gref)

    return
