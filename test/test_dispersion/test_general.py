"""
Testing dispersion module.
"""
from __future__ import annotations

import pytest
import torch

from dxtb.dispersion import new_dispersion
from dxtb.param import GFN1_XTB as par
from dxtb.utils import ParameterWarning


def test_none() -> None:
    dummy = torch.tensor(0.0)
    _par = par.copy(deep=True)

    with pytest.warns(ParameterWarning):
        _par.dispersion = None
        assert new_dispersion(dummy, _par) is None

        del _par.dispersion
        assert new_dispersion(dummy, _par) is None
