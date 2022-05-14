import pytest
import tad_dftd3 as d3
import torch

from xtbml.exlibs.tbmalt.batch import pack
from xtbml.typing import Tensor

from .samples import structures


class TestDispersion:
    """
    Testing dispersion energy and autodiff.
    These tests are taken from https://github.com/awvwgk/tad-dftd3/tree/main/tests and are only included for the sake of completeness.
    """

    @classmethod
    def setUpClass(cls):
        print(cls.__name__)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_disp_batch(self, dtype) -> None:
        sample1, sample2 = (
            structures["PbH4-BiH3"],
            structures["C6H5I-CH3SH"],
        )
        numbers: Tensor = pack(
            (
                sample1["numbers"],
                sample2["numbers"],
            )
        )
        positions: Tensor = pack(
            (
                sample1["positions"].type(dtype),
                sample2["positions"].type(dtype),
            )
        )
        c6: Tensor = pack(
            (
                sample1["c6"].type(dtype),
                sample2["c6"].type(dtype),
            )
        )
        rvdw = d3.data.vdw_rad_d3[numbers.unsqueeze(-1), numbers.unsqueeze(-2)]
        r4r2 = d3.data.sqrt_z_r4_over_r2[numbers]
        param = dict(a1=0.49484001, s8=0.78981345, a2=5.73083694)
        ref = torch.tensor(
            [
                [
                    -3.5479912602e-04,
                    -8.9124281989e-05,
                    -8.9124287363e-05,
                    -8.9124287363e-05,
                    -1.3686794039e-04,
                    -3.8805575850e-04,
                    -8.7387460069e-05,
                    -8.7387464149e-05,
                    -8.7387460069e-05,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                    -0.0000000000e-00,
                ],
                [
                    -4.1551151549e-04,
                    -3.9770287009e-04,
                    -4.1552470565e-04,
                    -4.4246829733e-04,
                    -4.7527776799e-04,
                    -4.4258484762e-04,
                    -1.0637547378e-03,
                    -1.5452322970e-04,
                    -1.9695663808e-04,
                    -1.6184434935e-04,
                    -1.9703176496e-04,
                    -1.6183339573e-04,
                    -4.6648977616e-04,
                    -1.3764556692e-04,
                    -2.4555353368e-04,
                    -1.3535967638e-04,
                    -1.5719227870e-04,
                    -1.1675684940e-04,
                ],
            ]
        ).type(dtype)

        energy = d3.disp.dispersion(
            numbers, positions, c6, rvdw, r4r2, d3.disp.rational_damping, **param
        )

        assert energy.dtype == dtype
        assert torch.allclose(energy, ref)

    @pytest.mark.grad
    def test_param_grad(self):
        dtype = torch.float64
        sample = structures["C4H5NCS"]
        numbers = sample["numbers"]
        positions = sample["positions"].type(dtype)
        param = (
            torch.tensor(1.00000000, requires_grad=True, dtype=dtype),
            torch.tensor(0.78981345, requires_grad=True, dtype=dtype),
            torch.tensor(0.49484001, requires_grad=True, dtype=dtype),
            torch.tensor(5.73083694, requires_grad=True, dtype=dtype),
        )
        label = ("s6", "s8", "a1", "a2")

        def func(*inputs):
            input_param = {label[i]: input for i, input in enumerate(inputs)}
            return d3.dftd3(numbers, positions, input_param)

        assert torch.autograd.gradcheck(func, param)
