import pytest
import torch

from xtbml.solvation import alpb

from .samples import mb16_43


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("sample", [mb16_43["01"], mb16_43["02"]])
def test_gb_single(dtype: torch.dtype, sample, dielectric_constant=78.9):

    dielectric_constant = torch.tensor(dielectric_constant, dtype=dtype)

    numbers = sample["numbers"]
    positions = sample["positions"].type(dtype)
    charges = sample["charges"].type(dtype)

    ihelp = "IndexHelper"
    gb = alpb.GeneralizedBorn(numbers, dielectric_constant)
    cache = gb.get_cache(numbers, positions, ihelp)
    torch.set_printoptions(precision=14)
    energies = gb.get_atom_energy(charges, ihelp, cache)
    print(energies)
    assert torch.allclose(energies, sample["energies"].type(dtype))
