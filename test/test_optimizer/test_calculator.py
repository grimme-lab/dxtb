import pytest
from pathlib import Path

import torch

from dxtb.data.dataset import SampleDataset
from dxtb.data.samples import Sample
from dxtb.optimizer.param_optim import ParameterOptimizer, reparametrise
from dxtb.xtb.calculator import Calculator
from dxtb.param.gfn1 import GFN1_XTB


class TestCalculator:
    """Test calculator for updating internal parameter."""

    def setup(self):
        # load data as batched sample
        path = Path(__file__).resolve().parents[0] / "samples.json"
        dataset = SampleDataset.from_json(path)
        self.batch = Sample.pack(dataset.samples)

    def test_caclulator_update(self):

        name = "hamiltonian.kcn"
        param = torch.tensor([1.0, 2.0, 3.0]).double()

        calc = Calculator(self.batch.numbers, self.batch.positions, GFN1_XTB)
        Calculator.set_param(calc, name, param)
        assert param.data_ptr() == Calculator.get_param(calc, name).data_ptr()
