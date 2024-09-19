# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test integral configuration.
"""

from __future__ import annotations

import pytest
import torch

from dxtb._src.constants import defaults, labels
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.typing import get_default_dtype
from dxtb.config import Config as Cfg

from ..conftest import DEVICE


def test_default() -> None:
    cfg = Cfg()
    assert cfg.strict == defaults.STRICT
    assert cfg.exclude == defaults.EXCLUDE
    assert cfg.method == defaults.METHOD
    assert cfg.grad == False
    assert cfg.batch_mode == defaults.BATCH_MODE

    assert cfg.ints.cutoff == defaults.INTCUTOFF
    assert cfg.ints.level == defaults.INTLEVEL
    assert cfg.ints.uplo == defaults.INTUPLO

    assert cfg.anomaly == False
    assert cfg.device == torch.device("cpu") if DEVICE is None else DEVICE
    assert cfg.dtype == get_default_dtype()

    assert cfg.scf.maxiter == defaults.MAXITER
    assert cfg.scf.mixer == defaults.MIXER
    assert cfg.scf.damp == defaults.DAMP
    assert cfg.scf.guess == defaults.GUESS
    assert cfg.scf.scf_mode == defaults.SCF_MODE
    assert cfg.scf.scp_mode == defaults.SCP_MODE
    assert cfg.scf.x_atol == defaults.X_ATOL
    assert cfg.scf.f_atol == defaults.F_ATOL
    assert cfg.scf.force_convergence == False

    assert cfg.scf.fermi.etemp == defaults.FERMI_ETEMP
    assert cfg.scf.fermi.maxiter == defaults.FERMI_MAXITER
    assert cfg.scf.fermi.thresh == defaults.FERMI_THRESH
    assert cfg.scf.fermi.partition == defaults.FERMI_PARTITION

    assert cfg.cache.enabled == defaults.CACHE_ENABLED
    assert cfg.cache.store.hcore == defaults.CACHE_STORE_HCORE
    assert cfg.cache.store.overlap == defaults.CACHE_STORE_OVERLAP
    assert cfg.cache.store.dipole == defaults.CACHE_STORE_DIPOLE
    assert cfg.cache.store.quadrupole == defaults.CACHE_STORE_QUADRUPOLE
    assert cfg.cache.store.charges == defaults.CACHE_STORE_CHARGES
    assert cfg.cache.store.coefficients == defaults.CACHE_STORE_COEFFICIENTS
    assert cfg.cache.store.density == defaults.CACHE_STORE_DENSITY
    assert cfg.cache.store.fock == defaults.CACHE_STORE_FOCK
    assert cfg.cache.store.iterations == defaults.CACHE_STORE_ITERATIONS
    assert cfg.cache.store.mo_energies == defaults.CACHE_STORE_MO_ENERGIES
    assert cfg.cache.store.occupation == defaults.CACHE_STORE_OCCUPATIONS
    assert cfg.cache.store.potential == defaults.CACHE_STORE_POTENTIAL

    assert cfg.max_element == defaults.MAX_ELEMENT

    if has_libcint is True:
        assert cfg.ints.driver == defaults.INTDRIVER
    else:
        assert cfg.ints.driver == labels.INTDRIVER_ANALYTICAL


def test_method() -> None:
    cfg = Cfg(method=labels.GFN1_XTB_STRS[0])
    assert cfg.method == labels.GFN1_XTB

    cfg = Cfg(method=labels.GFN1_XTB)
    assert cfg.method == labels.GFN1_XTB

    if has_libcint is True:
        cfg = Cfg(method=labels.GFN2_XTB)
        assert cfg.method == labels.GFN2_XTB

        cfg = Cfg(method=labels.GFN2_XTB_STRS[0])
        assert cfg.method == labels.GFN2_XTB
    else:
        with pytest.raises(RuntimeError):
            Cfg(method=labels.GFN2_XTB_STRS[0])

        with pytest.raises(RuntimeError):
            Cfg(method=labels.GFN2_XTB)


def test_method_fail() -> None:
    with pytest.raises(ValueError):
        Cfg(method="invalid")

    with pytest.raises(ValueError):
        Cfg(method=-999)

    with pytest.raises(TypeError):
        Cfg(method=1.0)  # type: ignore


def test_fail_incompatibility() -> None:
    with pytest.raises(RuntimeError):
        Cfg(method=labels.GFN2_XTB, int_driver=labels.INTDRIVER_LEGACY)
