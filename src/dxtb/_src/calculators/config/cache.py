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
Integral configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from dxtb._src.constants import defaults

__all__ = ["ConfigCache"]


@dataclass
class ConfigCacheStore:
    hcore: bool
    """Whether to store the core Hamiltonian matrix."""
    overlap: bool
    """Whether to store the overlap matrix."""
    dipole: bool
    """Whether to store the dipole moment."""
    quadrupole: bool
    """Whether to store the quadrupole moment."""
    #
    charges: bool
    """Whether to store the atomic charges."""
    coefficients: bool
    """Whether to store the MO coefficients."""
    density: bool
    """Whether to store the density matrix."""
    fock: bool
    """Whether to store the Fock matrix."""
    iterations: bool
    """Whether to store the number of SCF iterations."""
    mo_energies: bool
    """Whether to store the MO energies."""
    occupation: bool
    """Whether to store the occupation numbers."""
    potential: bool
    """Whether to store the potential matrix."""


class ConfigCache:
    """
    Configuration for the calculator cache.

    All configuration options are represented as integers. String options are
    converted to integers in the constructor.
    """

    enabled: bool
    """Enable or disable the cache."""

    store: ConfigCacheStore
    """Container for which quantities to store."""

    def __init__(
        self,
        *,
        enabled: bool = defaults.CACHE_ENABLED,
        #
        hcore: bool = defaults.CACHE_STORE_HCORE,
        overlap: bool = defaults.CACHE_STORE_OVERLAP,
        dipole: bool = defaults.CACHE_STORE_DIPOLE,
        quadrupole: bool = defaults.CACHE_STORE_QUADRUPOLE,
        #
        charges: bool = defaults.CACHE_STORE_CHARGES,
        coefficients: bool = defaults.CACHE_STORE_COEFFICIENTS,
        density: bool = defaults.CACHE_STORE_DENSITY,
        fock: bool = defaults.CACHE_STORE_FOCK,
        iterations: bool = defaults.CACHE_STORE_ITERATIONS,
        mo_energies: bool = defaults.CACHE_STORE_MO_ENERGIES,
        occupation: bool = defaults.CACHE_STORE_OCCUPATIONS,
        potential: bool = defaults.CACHE_STORE_POTENTIAL,
    ) -> None:
        self.enabled = enabled

        self.store = ConfigCacheStore(
            hcore=hcore,
            overlap=overlap,
            dipole=dipole,
            quadrupole=quadrupole,
            #
            charges=charges,
            coefficients=coefficients,
            density=density,
            fock=fock,
            iterations=iterations,
            mo_energies=mo_energies,
            occupation=occupation,
            potential=potential,
        )
