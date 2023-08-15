"""
Testing overlap gradient (autodiff).
"""
from __future__ import annotations

import pytest
import torch

from dxtb._types import DD, Tensor
from dxtb.basis import Basis, IndexHelper
from dxtb.integral import libcint as intor
from dxtb.param import GFN1_XTB as par
from dxtb.param import get_elem_angular
from dxtb.utils import batch, is_basis_list

from ..utils import dgradcheck, dgradgradcheck
from .samples import samples

sample_list = ["H2", "HHe", "LiH", "Li2", "S2", "H2O", "SiH4"]

# FIXME: Investigate low tolerance (normally 1e-7)!
tol = 1e-7

device = None


def num_grad(
    numbers: Tensor, ihelp: IndexHelper, positions: Tensor, intstr: str
) -> Tensor:
    # setup numerical gradient
    positions = positions.detach().clone()

    norb = int(ihelp.orbitals_per_shell.sum())
    gradient = torch.zeros(
        (3, 3, norb, norb), dtype=positions.dtype, device=positions.device
    )
    step = 1.0e-5

    def compute_integral(pos: torch.Tensor) -> torch.Tensor:
        bas = Basis(numbers, par, ihelp, dtype=positions.dtype, device=positions.device)
        atombases = bas.create_dqc(pos)
        assert is_basis_list(atombases)

        wrapper = intor.LibcintWrapper(atombases, ihelp)
        return intor.int1e(intstr, wrapper)

    # Loop over all atoms and their x, y, z coordinates
    for atom in range(positions.shape[0] - 1):
        print(atom, positions.shape)
        for direction in range(3):
            positions[atom, direction] += step
            ir = compute_integral(positions)

            positions[atom, direction] -= 2 * step
            il = compute_integral(positions)

            positions[atom, direction] += step
            gradient[direction] += 0.5 * (ir - il) / step

    print("")
    print("")
    print("")
    return gradient


@pytest.mark.grad
@pytest.mark.parametrize("dtype", [torch.double])
@pytest.mark.parametrize("name", sample_list)
def test_grad(dtype: torch.dtype, name: str):
    """Prepare gradient check from `torch.autograd`."""
    dd: DD = {"device": device, "dtype": dtype}

    sample = samples[name]
    numbers = sample["numbers"].to(device)
    positions = sample["positions"].to(**dd)
    positions[0] = torch.tensor([0, 0, 0])
    print(positions)
    positions.requires_grad_(True)

    ihelp = IndexHelper.from_numbers(numbers, get_elem_angular(par.element))
    bas = Basis(numbers, par, ihelp, **dd)

    atombases = bas.create_dqc(positions)
    assert is_basis_list(atombases)

    INTSTR = "r0"

    wrapper = intor.LibcintWrapper(atombases, ihelp)
    i = intor.int1e(INTSTR, wrapper)
    print()
    igrad = intor.int1e(f"ip{INTSTR}", wrapper)
    igrad = igrad + igrad.mT
    print("igrad\n", igrad)
    # assert False

    print(igrad.shape)
    numgrad = num_grad(numbers, ihelp, positions, INTSTR)
    print("numgrad\n", numgrad)
    print("")
    print("")
    print("diff")
    print(igrad + numgrad)