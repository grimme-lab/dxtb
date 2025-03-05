#!/usr/bin/env python3
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
https://github.com/grimme-lab/dxtb/issues/206
"""
import concurrent.futures
import tempfile
from tempfile import NamedTemporaryFile

import torch
from tad_mctc.convert import numpy_to_tensor, tensor_to_numpy
from tad_mctc.units import AA2AU

from dxtb import GFN1_XTB
from dxtb import Calculator as DxtbCalculator
from dxtb import kill_timer

try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator as AseCalculator
    from ase.io import read
    from ase.optimize import BFGS
except ImportError:
    raise SystemExit("Please install ASE to run this example.")

kill_timer()

# device should be a string!!
DEVICE = "cpu"
dd = {"device": torch.device(DEVICE), "dtype": torch.double}


class DxtbAseCalculator(AseCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        parametrization=GFN1_XTB,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.parametrization = parametrization

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties=["energy"],
        system_changes=None,
    ):
        AseCalculator.calculate(self, atoms, properties, system_changes)
        assert atoms is not None

        numbers = numpy_to_tensor(
            atoms.get_atomic_numbers(), device=dd["device"]
        )
        positions = numpy_to_tensor(atoms.get_positions() * AA2AU, **dd)
        chrg = atoms.info["charge"]

        dxtb_calculator = DxtbCalculator(
            numbers=numbers,
            par=self.parametrization,
            opts={"cache_enabled": False, "verbosity": 0},
            **dd,
        )

        energy = dxtb_calculator.get_energy(positions, chrg=chrg)

        # The calculator was not reset here!
        positions.requires_grad_(True)
        forces = dxtb_calculator.get_forces(positions, chrg=chrg)

        self.results.update(
            {
                "energy": tensor_to_numpy(energy),
                "forces": tensor_to_numpy(forces),
            }
        )


def optimize_geometry(xyz_content: str):
    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w+") as temp_input:
        temp_input.write(xyz_content)
        temp_input.flush()

        atoms = read(filename=temp_input.name)
        assert not isinstance(atoms, list)
        atoms.info.update({"charge": 0})

        with NamedTemporaryFile(
            "w+", suffix=".traj", delete=False
        ) as temp_traj:
            atoms.calc = DxtbAseCalculator(parametrization=GFN1_XTB)
            dyn = BFGS(atoms, trajectory=temp_traj.name)
            dyn.run(fmax=0.05, steps=500)
            return atoms


def main() -> int:
    xyz_1 = """3

    C      1.394181    3.856774   -1.611889
    O      2.232295    3.058194   -2.123577
    O      0.256209    3.513960   -1.175374
    """

    xyz_2 = """3

    C      1.394181    3.856774   -1.611889
    O      2.232295    3.058194   -2.123577
    O      0.256209    3.513960   -1.175374
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(optimize_geometry, sample)
            for sample in [xyz_1, xyz_2]
        ]
        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]

    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
