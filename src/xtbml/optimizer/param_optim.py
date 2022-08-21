from __future__ import annotations
from typing import Callable, List
import torch
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from tad_dftd3 import dftd3

from xtbml.param.base import Param

from ..typing import Tensor
from ..data.dataset import SampleDataset
from ..data.samples import Sample
from ..xtb.calculator import Calculator
from ..param.gfn1 import GFN1_XTB
from ..param.base import Param
from ..constants import AU2KCAL

from ..utils.utils import rgetattr, rsetattr


def convert_to_tensor(
    params: Param,
    name: str,
    requires_grad: bool = True,
    dtype: torch.dtype = torch.float64,
) -> Tensor:

    # TODO: check that converting entries to tensors does not break anything
    def convert(x: int | float) -> Tensor:
        print("convert ", type(x))
        assert isinstance(x, int) or isinstance(x, float), f"Of type: {type(x)}"
        return torch.tensor(x, requires_grad=requires_grad, dtype=dtype)

    # for dict entries allow to split name by dict key
    # i.e. name= "hamiltonian.xtb.kpair['Pt-Ti']" ->['Pt-Ti'] as key
    key = None
    split = name.split("[")
    if len(split) > 1:
        name, key = split
        for s in ["'", '"', "]"]:
            key = key.replace(s, "")
        v = rgetattr(params, name)
        v[key] = convert(v[key])
        rev = v[key]
    elif len(split) > 2:
        raise AttributeError
    else:
        v = convert(rgetattr(params, name))
        rev = v

    rsetattr(params, name, v)

    return rev


class ParameterOptimizer(nn.Module):
    """Pytorch model for gradient optimization of given forward() function."""

    def __init__(self, parametrisation: Param, names: list[str]):
        super().__init__()

        # don't mess with global variables
        assert id(parametrisation) != id(GFN1_XTB)

        # parametrisation as proprietary attribute
        self.parametrisation = parametrisation

        # TODO: better naming for parametrisation and parameters

        abc = [convert_to_tensor(self.parametrisation, name) for name in names]

        # register as learnable model parameters
        self.params = nn.ParameterList(
            [nn.Parameter(i) for i in abc]
            # [
            #     # convert all trainable parameters to tensors
            #     nn.Parameter(convert_to_tensor(self.parametrisation, name))
            #     for name in names
            # ]
        )

        # TODO: write a getter/setter function for Params object
        # TODO: check that params are updated in self.parametrisation
        # TODO: check that params are not overwritten in Calculator setup (i.e. floats converted to tensors)
        import sys

        # sys.exit(0)

    def forward(self, x: Sample):
        """
        Function to be optimised.
        """

        positions = x.positions.detach()
        positions = positions.double()
        positions.requires_grad_(True)

        calc = Calculator(x.numbers, positions, self.parametrisation)

        # # assign model weights (= learnable parameter) to calculator
        # for i, param in enumerate(self.params):
        #     Calculator.set_param(calc, self.params_name[i], self.params[i])

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
            "s6": self.parametrisation.dispersion.d3.s6,
            "s8": self.parametrisation.dispersion.d3.s8,
            "a1": self.parametrisation.dispersion.d3.a1,
            "a2": self.parametrisation.dispersion.d3.a2,
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
        if verbose:
            print(f"epoch {i}")
        for batch in dataloader:
            if verbose:
                print(f"batch: {batch}")
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
            root.glob("samples_DUMMY.json")
            # root.glob("samples_HCNO_tiny.json")
        )  # samples_*.json samples_HE*.json samples_DUMMY.json
        return SampleDataset.from_json(list_of_path)

    # load data as batched sample
    path = Path(__file__).resolve().parents[3] / "data" / "PTB"
    dataset = get_ptb_dataset(path)
    batch = Sample.pack(dataset.samples)
    print("Number samples", len(dataset))

    # dynamic packing for each batch
    dataloader = DataLoader(
        dataset,
        batch_size=6,
        shuffle=False,
        collate_fn=Sample.pack,
        drop_last=False,
    )

    # setup calculator and get parameters
    calc = Calculator(batch.numbers, batch.positions, GFN1_XTB)
    # names = ["hamiltonian.kcn", "hamiltonian.hscale", "hamiltonian.shpoly"]
    names = ["hamiltonian.kpair"]
    params_to_opt = [Calculator.get_param(calc, name) for name in names]
    # TODO: check dataloader is setup for _all_ elements in samples, during batch only limited might occur
    #       --> does that work with calculator internal unique element representation in batch?

    # TODO: better to add attributes directly to Param toml object (GFN1_XTB)
    #       -- then Calculator constructor handles the correct assignment

    names = ["hamiltonian.xtb.kpair"]
    names = [
        "hamiltonian.xtb.kpair['H-H']",
        "hamiltonian.xtb.kpair['N-H']",
    ]
    # TODO: get all dict entries as strings and add to list

    print(batch.numbers)
    # return

    """print("GFN1_XTB.meta", GFN1_XTB.meta)
    print("GFN1_XTB.element", GFN1_XTB.element)
    print("GFN1_XTB.hamiltonian", GFN1_XTB.hamiltonian)
    print("GFN1_XTB.dispersion", GFN1_XTB.dispersion)
    print("GFN1_XTB.repulsion", GFN1_XTB.repulsion)
    print("GFN1_XTB.charge", GFN1_XTB.charge)
    print("GFN1_XTB.multipole", GFN1_XTB.multipole)
    print("GFN1_XTB.halogen", GFN1_XTB.halogen)
    print("GFN1_XTB.thirdorder", GFN1_XTB.thirdorder)"""

    print("add stuff directly to GFN1_XTB param file")
    # return

    # setup model, optimizer and loss function
    model = ParameterOptimizer(GFN1_XTB.copy(deep=True), names)
    # model = ParameterOptimizer(params_to_opt, names)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = torch.nn.MSELoss(reduction="sum")
    # optmizier options: Adagrad, RMSPROP

    """epoch 4
    tensor(0.3194, dtype=torch.float64, grad_fn=<MseLossBackward0>)
    FINAL model.params ParameterList(  (0): Parameter containing: [torch.DoubleTensor of size 5x5])
    Parameter containing:
    tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
            [1.0000, 1.1010, 1.0620, 0.8050, 0.7036],
            [1.0000, 1.0620, 0.5571, 1.0000, 0.8423],
            [1.0000, 0.8050, 1.0000, 0.6918, 1.0000],
            [1.0000, 0.7036, 0.8423, 1.0000, 1.3667]], dtype=torch.float64, requires_grad=True)"""

    print("INITAL model.params", model.params)
    for param in model.params:
        print(param)

    losses = training_loop(model, opt, loss_fn, dataloader, n=5)

    print("FINAL model.params", model.params)
    for param in model.params:
        print(param)

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
