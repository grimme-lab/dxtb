import pytest
import torch
import math

from xtbml import utils
from xtbml.basis import IndexHelper
from xtbml.exlibs.tbmalt import batch, Geometry
from xtbml.param import GFN1_XTB as par, get_element_angular
from xtbml.xtb.calculator import Calculator

from .samples import mb16_43, dimers


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("testcase",
    [
        # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
        (mb16_43["H2"], -1.0585984032484),
        (mb16_43["LiH"], -0.88306406116865),
        (mb16_43["SiH4"], -4.0384093532453),
    ]
)
def test_single(testcase, dtype):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    sample, ref = testcase

    mol = Geometry(sample["numbers"], sample["positions"].type(dtype))

    calc = Calculator(mol, par)
    ihelp = IndexHelper.from_numbers(mol.atomic_numbers, get_element_angular(par.element))

    results = calc.singlepoint(mol, ihelp)
    assert pytest.approx(ref, abs=tol) == results.get("energy").sum(-1).item()


@pytest.mark.parametrize("testcase",
    [
        # Values obtain with tblite 0.2.1 disabling repulsion and dispersion
        (mb16_43["S2"], -7.3285116888517),
        (dimers["PbH4-BiH3"], -7.6074262079844),
        (dimers["C6H5I-CH3SH"], -27.612142805843),
    ]
)
def test_single2(testcase, dtype=torch.float):
    tol = math.sqrt(torch.finfo(dtype).eps) * 10

    sample, ref = testcase

    mol = Geometry(sample["numbers"], sample["positions"].type(dtype))

    calc = Calculator(mol, par)
    ihelp = IndexHelper.from_numbers(mol.atomic_numbers, get_element_angular(par.element))

    results = calc.singlepoint(mol, ihelp)
    assert pytest.approx(ref, abs=tol) == results.get("energy").sum(-1).item()
