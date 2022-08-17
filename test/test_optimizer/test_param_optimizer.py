import pytest
from pathlib import Path

import torch

from xtbml.data.dataset import SampleDataset
from xtbml.data.samples import Sample
from xtbml.optimizer.param_optim import ParameterOptimizer, training_loop
from xtbml.xtb.calculator import Calculator
from xtbml.param.gfn1 import GFN1_XTB


class TestParameterOptimizer:
    """Test the parameter optimizer functionality, especially
    in the context of re-parametrising the GFN-xTB parametrisation."""

    @classmethod
    def setup_class(cls):
        print(f"\n{cls.__name__}")

    def setup(self):
        # load data as batched sample
        path = Path(__file__).resolve().parents[0] / "samples.json"
        dataset = SampleDataset.from_json(path)
        self.batch = Sample.pack(dataset.samples)

        self.loss_fn = torch.nn.MSELoss(reduction="sum")

    def optimisation(self, model):
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        training_loop(model, opt, self.loss_fn, self.batch, n=20)

    def test_simple_param_optimisation(self):

        # get parameter to optimise
        param_to_opt = torch.tensor([0.6])
        param_to_opt = param_to_opt.double().requires_grad_(True)
        name = "Dummy"

        batch = self.batch

        class SimpleParameterOptimizer(ParameterOptimizer):
            def calc_gradient(self, energy, positions):
                # scaling the reference gradient
                gradient = self.weights * batch.gref
                return gradient

        # optimise xtb
        model = SimpleParameterOptimizer(param_to_opt, name)
        self.optimisation(model)
        # print("FINAL model.weights", model.weights)

        # should converge to 1.0
        assert param_to_opt.data_ptr() == model.weights.data_ptr()
        assert torch.isclose(param_to_opt, torch.tensor(1.0).double(), rtol=0.1)

    def test_param_optimisation(self):

        # get parameter to optimise
        calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        name = "hamiltonian.shpoly"
        param_to_opt = Calculator.get_param(calc, name)

        # optimise xtb
        model = ParameterOptimizer(param_to_opt, name)
        self.optimisation(model)

        # inference
        prediction, grad = model(self.batch)
        lopt = self.loss_fn(grad, self.batch.gref)

        assert torch.isclose(lopt, torch.tensor(0.0006, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(
            prediction,
            torch.tensor([[-5.0109, -0.3922, -0.3922]], dtype=torch.float64),
            atol=1e-4,
        )

    def test_loss_backward(self):

        # get parameter to optimise
        calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        name = "hamiltonian.shpoly"
        param_to_opt = Calculator.get_param(calc, name)

        model = ParameterOptimizer(param_to_opt, name)
        _, grad = model(self.batch)
        y_true = self.batch.gref.double()
        loss = self.loss_fn(grad, y_true)

        # gradients via loss backward
        loss.backward(inputs=model.weights)
        grad_bw = model.weights.grad

        _, grad = model(self.batch)
        loss = self.loss_fn(grad, y_true)

        # gradients via (manual) autograd
        grad_ad = torch.autograd.grad(
            loss,
            model.weights,
        )[0]

        assert torch.allclose(
            grad_bw,
            grad_ad,
            atol=1e-4,
        )
