import pytest
from pathlib import Path
import tomli as toml

import torch

from dxtb.data.dataset import SampleDataset
from dxtb.data.samples import Sample
from dxtb.optimizer.param_optim import ParameterOptimizer, train
from dxtb.xtb.calculator import Calculator
from dxtb.param.gfn1 import GFN1_XTB
from dxtb.param.base import Param


class TestParameterOptimizer:
    """Test the parameter optimizer functionality, especially
    in the context of re-parametrising the GFN-xTB parametrisation."""

    def setup(self):
        # load data as batched sample
        path = Path(__file__).resolve().parents[0] / "samples.json"
        dataset = SampleDataset.from_json(path)
        self.batch = Sample.pack(dataset.samples)

        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=Sample.pack,
            drop_last=False,
        )
        self.loss_fn = torch.nn.MSELoss(reduction="sum")

        # parameter to optimise
        with open(
            #Path(__file__).resolve().parents[0] / "parametrisation.toml", "rb"
            Path(__file__).resolve().parents[0] / "gfn1-xtb.toml", "rb"
        ) as fd:
            self.param = Param(**toml.load(fd))

    def optimisation(self, model, epochs=1):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        return train(model, self.dataloader, optimizer, self.loss_fn, epochs)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_optimisation_by_identity(self, dtype: torch.dtype):

        # set dummy parameter
        name = "hamiltonian.xtb.enscale"
        value = 123.0
        self.param.set_param(name, value)
        fake_reference = self.batch.gref

        class ParameterOptimizerByIdentity(ParameterOptimizer):
            def calc_gradient(self, energy, positions):
                # scaling the reference gradient
                gradient = self.params[0] * fake_reference
                return gradient

        # optimise xtb
        model = ParameterOptimizerByIdentity(self.param, [name], dtype=dtype)
        self.optimisation(model)

        res = self.param.get_param(name)

        # should converge to identical value as input
        assert res.data_ptr() == model.params[0].data_ptr()
        assert torch.isclose(res, torch.tensor(value, dtype=dtype), atol=0.1)

    @pytest.mark.parametrize("dtype", [torch.float32])#, torch.float64
    def test_param_optimisation(self, dtype: torch.dtype):

        # AUTOGRAD RuntimeError: Function 'DivBackward0' returned nan values in its 0th output.
        print("start")
        with open(
            Path(__file__).resolve().parents[0] / "gfn1-xtb.toml", "rb"
        ) as fd:
            parameters = Param(**toml.load(fd))

        print(self.batch)

        torch.autograd.set_detect_anomaly(True)

        name = "hamiltonian.xtb.enscale"
        model = ParameterOptimizer(parameters, [name], dtype=dtype)
        loss_fn = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        optimizer.zero_grad(set_to_none=False)

        print("Params before", model.params[0])

        energy, grad = model(self.batch)
        print("energy", energy)
        print("grad", grad)
        print("grad.sum", grad.sum())
        print("grad_ref", self.batch.gref)
        loss = loss_fn(grad, self.batch.gref.type(model.dtype))
        
        loss.backward(inputs=model.params)
        #energy.backward(inputs=model.params)
        
        print("Params", model.params[0])
        print("Param grads", model.params[0].grad)
        optimizer.step() # updating value to nan

        print("Params after", model.params[0])

        return

        torch.autograd.anomaly_mode.set_detect_anomaly(True)

        with open(
            Path(__file__).resolve().parents[0] / "gfn1-xtb.toml", "rb"
        ) as fd:
            self.param = Param(**toml.load(fd))

        # set dummy parameter
        name = "hamiltonian.xtb.kpol" #enscale
        value = -0.007 # for 123 runs into Nan
        #self.param.set_param(name, value)

        # # get parameter to optimise
        # calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        # name = "hamiltonian.shpoly"
        # self.param = Calculator.get_param(calc, name)

        print('START')
        print("init", self.param.get_param(name))

        model = ParameterOptimizer(self.param, [name], dtype=dtype)
        # assert torch.equal(self.param.get_param(name), torch.tensor(value, dtype=dtype, requires_grad=True)), "Parameters not converted correctly inside ParameterOptimizer constructor."
        # assert torch.equal(self.param.get_param(name), model.params[0]), "Parameters not correctly added to model params."
        
        self.optimisation(model)

        res = self.param.get_param(name)
        print("res", res)


        return

        # optimise xtb
        model = ParameterOptimizer(self.param, [name], dtype=dtype)
        self.optimisation(model)

        # inference
        prediction, grad = model(self.batch)
        loss = self.loss_fn(grad, self.batch.gref)

        res = self.param.get_param(name)

        print("Loss", loss)
        print(grad, self.batch.gref)
        print("res", res)

        assert torch.isclose(loss, torch.tensor(0.0006, dtype=torch.float64), atol=1e-5)
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
        )

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
