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

        self.weights_name = weights_name

    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        # TODO: optional detaching
        positions = x.positions  # x.positions.detach()
        positions = positions.double()
        positions.requires_grad_(True)

        calc = Calculator(x.numbers, positions, GFN1_XTB)

        # assign model weights (= learnable parameter) to calculator
        Calculator.set_param(calc, self.weights_name, self.weights)
        # calc.hamiltonian.shpoly = self.weights
        # TODO: implement this flexibly for all parameter

        """print("investigate calc update")
        print(self.weights)
        print(calc.hamiltonian.shpoly)
        print(calc.hamiltonian.shpoly.dtype)
        print(self.weights.data_ptr() == calc.hamiltonian.shpoly.data_ptr())
        print("tensors should not be copied but moved")"""
        assert self.weights.data_ptr() == calc.hamiltonian.shpoly.data_ptr()

        # loss
        # tensor(8.1559e-06)
        # tensor(1.0621e-07, dtype=torch.float64, grad_fn=<MseLossBackward0>)
        # prediction
        # tensor([[-3052.8406,  -283.5126,  -283.5126]])
        # tensor([[-4.9355, -0.4757, -0.4757]], dtype=torch.float64, grad_fn=<AddBackward0>)
        # tensor([-10837.7598])

        # singlepoint calculation
        results = calc.singlepoint(x.numbers, positions, x.charges, verbosity=0)

        energy = results["energy"]

        # convert to kcal
        # energy = energy * AU2KCAL
        # TODO: why does this conversion change the training outcome of the gradient?
        #   could it be that the gradient in JSON is not converted to kcal? (i.e. |dE/dxyz|)

        gradient = torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=True,
        )[0]

        # positions.requires_grad_(False)
        # NOTE: this is not necessary, as long as loss backward is applied to model params

        ##
        # gradient = self.weights * x.gref
        # TODO: nice test with
        # param_to_opt = torch.tensor([0.2])
        #  (should converge to 1.0)
        ##

        return energy, gradient


def training_loop(model, optimizer, loss_fn, x, n=100):
    "Training loop for torch model."
    losses = []

    for i in range(n):
        print(f"epoch {i}")
        optimizer.zero_grad()
        energy, grad = model(x)
        y_true = x.gref.double()
        loss = loss_fn(grad, y_true)
        # alternative: Jacobian of loss

        # TODO: write test dldh == loss.backward(inputs=model.weights)
        """dldh = torch.autograd.grad(
            loss,
            model.weights,
        )[0]"""

        loss.backward(inputs=model.weights)
        # print("grad after backward", model.weights.grad)

        optimizer.step()
        losses.append(loss)
        print(loss)
    return losses


def example():
    def get_ptb_dataset(root: Path) -> SampleDataset:
        list_of_path = sorted(
            root.glob("samples_DUMMY*.json")
        )  # samples_*.json samples_HE*.json samples_DUMMY*.json
        return SampleDataset.from_json(list_of_path)

    # load data as batched sample
    path = Path(__file__).resolve().parents[3] / "data" / "PTB"
    dataset = get_ptb_dataset(path)
    batch = Sample.pack(dataset.samples[:1])
    # TODO: currently batch SCF not working (due to batching?)
    #       - check charge(?) batching with einsum mapping in SCF
    print(type(batch))
    print(batch.numbers.shape)

    # setup calculator
    calc = Calculator(batch.numbers, batch.positions, GFN1_XTB)

    name = "hamiltonian.shpoly"

    # TODO: set these weights from calculator object
    param_to_opt = Calculator.get_param(calc, name)
    # param_to_opt = shpoly
    # param_to_opt = torch.tensor([0.6])

    # setup model, optimizer and loss function
    m = ParameterOptimizer(param_to_opt, name)
    opt = torch.optim.Adam(m.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction="sum")

    print("INITAL model.weights", m.weights)

    losses = training_loop(m, opt, loss_fn, batch)

    print("FINAL model.weights", m.weights)

    # inference
    preds, grad = m(batch)

    print("loss")
    # TODO: do we have GFN2 values?
    lgfn1 = loss_fn(batch.ggfn1, batch.gref)
    lopt = loss_fn(grad, batch.gref)
    print(lgfn1)
    print(lopt)

    print("prediction")
    print(batch.egfn1)
    print(preds)
    print(batch.eref)

    return
