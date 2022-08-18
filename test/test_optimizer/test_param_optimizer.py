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

    """def test_simple_param_optimisation(self):

        # get parameter to optimise
        param_to_opt = torch.tensor([0.6])
        param_to_opt = param_to_opt.double().requires_grad_(True)
        name = "Dummy"

        batch = self.batch

        class SimpleParameterOptimizer(ParameterOptimizer):
            def calc_gradient(self, energy, positions):
                # scaling the reference gradient
                gradient = self.params[0] * batch.gref
                return gradient

        # optimise xtb
        model = SimpleParameterOptimizer([param_to_opt], [name])
        self.optimisation(model)
        # print("FINAL model.params", model.params)

        # should converge to 1.0
        assert param_to_opt.data_ptr() == model.params[0].data_ptr()
        assert torch.isclose(param_to_opt, torch.tensor(1.0).double(), rtol=0.1)

    def test_param_optimisation(self):

        # get parameter to optimise
        calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        name = "hamiltonian.shpoly"
        param_to_opt = Calculator.get_param(calc, name)

        # optimise xtb
        model = ParameterOptimizer([param_to_opt], [name])
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

        model = ParameterOptimizer([param_to_opt], [name])
        _, grad = model(self.batch)
        y_true = self.batch.gref.double()
        loss = self.loss_fn(grad, y_true)

        # gradients via loss backward
        loss.backward(inputs=model.params)
        grad_bw = model.params[0].grad

        _, grad = model(self.batch)
        loss = self.loss_fn(grad, y_true)

        # gradients via (manual) autograd
        grad_ad = torch.autograd.grad(
            loss,
            model.params,
        )[0]

        assert torch.allclose(
            grad_bw,
            grad_ad,
            atol=1e-4,
        )"""

    def test_multiple_param_optimisation(self):

        # get parameter to optimise
        calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        names = ["hamiltonian.kcn", "hamiltonian.hscale"]
        params_to_opt = [Calculator.get_param(calc, name) for name in names]

        # optimise xtb
        model = ParameterOptimizer(params_to_opt, names)
        self.optimisation(model)

        # inference
        prediction, grad = model(self.batch)
        lopt = self.loss_fn(grad, self.batch.gref)

        assert torch.isclose(
            lopt, torch.tensor(9.3381e-06, dtype=torch.float64), atol=1e-5
        )
        assert torch.allclose(
            model.params[0],
            torch.tensor([-0.3844, -0.0527, 0.3224, -0.2267], dtype=torch.float64),
            atol=1e-4,
        )
        assert torch.allclose(
            model.params[1],
            torch.tensor(
                [
                    [2.1530, 2.5532, 1.7393, 1.6536],
                    [2.5532, 2.6529, 2.4780, 2.8068],
                    [1.7393, 2.4780, 1.8500, 2.0800],
                    [1.6536, 2.8068, 2.0800, 2.2500],
                ],
                dtype=torch.float64,
            ),
            atol=1e-4,
        )
