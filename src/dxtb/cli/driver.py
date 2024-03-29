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
Driver class for running dxtb.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path

import torch
from tad_mctc import units
from tad_mctc.batch import pack
from tad_mctc.io import read

from .. import io
from ..components.interactions.external import new_efield
from ..config import Config
from ..constants import labels
from ..timing import timer
from ..xtb import Calculator, Result

logger = logging.getLogger(__name__)

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


def print_grad(grad, numbers) -> None:
    from tad_mctc.data import pse

    print("************************Gradient************************")
    print("")

    # Iterate over the tensor and print
    for i, row in enumerate(grad):
        # Get the atomic symbol corresponding to the atomic number
        symbol = pse.Z2S.get(int(numbers[i].item()), "?")
        print(f"{i+1:>3} {symbol:<2} : {row[0]:>15.9f} {row[1]:>15.9f} {row[2]:>15.9f}")


class Driver:
    """
    Driver class for running dxtb.
    """

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.batched = True
        self.base = self._set_base()
        self.chrg = self._set_attr("chrg")
        self.spin = self._set_attr("spin")

    def _set_base(self) -> list[Path]:
        """
        Set the base path from `args.file`.

        Returns
        -------
        Path
            Parent directory of given coordinate file.
        """
        return [Path(f).resolve().parent for f in self.args.file]

    def _set_attr(self, attr: str) -> int | list[int]:
        # set charge to input from Namespace
        val = getattr(self.args, attr)

        if isinstance(val, list):
            raise NotImplementedError("Setting charges as list not working.")

        # only search for file if not specified
        if val is not None:
            return val

        vals = []
        for path in self.base:
            # use charge (or spin) from file or set to zero
            if Path(path, FILES[attr]).is_file():
                vals.append(io.read_chrg(Path(path, FILES[attr])))
            else:
                vals.append(0)

        # do not return list if only one system is considered
        if len(vals) == 1:
            vals = vals[0]

        return vals

    def singlepoint(self) -> Result:
        timer.start("Setup")

        args = self.args

        # logging.basicConfig(
        # level=args.loglevel.upper(),
        # format="%(asctime)s %(levelname)s %(name)s::%(funcName)s -> %(message)s",
        # datefmt="%Y-%m-%d %H:%M:%S",
        # )
        # config = io.logutils.get_logging_config(level=args.loglevel.upper())
        # logging.basicConfig(**config)

        # args.verbosity contains default if not set, v/s are zero
        io.OutputHandler.verbosity = args.verbosity + args.v - args.s

        # setup output: streams, verbosity
        if args.json:
            io.OutputHandler.setup_json_logger()

        io.OutputHandler.header()
        io.OutputHandler.sysinfo()

        if args.detect_anomaly:
            # pylint: disable=import-outside-toplevel
            from torch.autograd.anomaly_mode import set_detect_anomaly

            set_detect_anomaly(True)

        dd = {"device": args.device, "dtype": args.dtype}

        opts = {
            "spin": args.spin,
        }

        config = Config.from_args(args)

        io.OutputHandler.write(config.info())

        if len(args.file) > 1:
            _n, _p = zip(
                *[read.read_from_path(f, args.filetype, **dd) for f in args.file]
            )
            numbers = pack(_n)
            positions = pack(_p)
        else:
            _n, _p = io.read_structure_from_file(args.file[0], args.filetype)
            numbers = torch.tensor(_n, dtype=torch.long, device=dd["device"])
            positions = torch.tensor(_p, **dd)

        chrg = torch.tensor(self.chrg, **dd)

        if args.grad is True:
            positions.requires_grad = True

        if args.method.lower() == "gfn1" or args.method.lower() == "gfn1-xtb":
            # pylint: disable=import-outside-toplevel
            from dxtb.param import GFN1_XTB as par
        elif args.method.lower() == "gfn2" or args.method.lower() == "gfn2-xtb":
            raise NotImplementedError("GFN2-xTB is not implemented yet.")
        else:
            raise ValueError(f"Unknown method '{args.method}'.")

        # INTERACTIONS
        interactions = []

        needs_field = any(
            [
                args.ir,
                args.ir_numerical,
                args.raman,
                args.raman_numerical,
                args.dipole,
                args.dipole_numerical,
                args.polarizability,
                args.polarizability_numerical,
                args.hyperpolarizability,
                args.hyperpolarizability_numerical,
            ]
        )

        if args.efield is not None or needs_field is True:
            field = (
                torch.tensor(
                    args.efield if args.efield is not None else [0, 0, 0],
                    **dd,
                    requires_grad=needs_field,
                )
                * units.VAA2AU
            )
            interactions.append(new_efield(field, **dd))

        # setup calculator
        calc = Calculator(numbers, par, opts=config, interaction=interactions, **dd)
        timer.stop("Setup")

        ####################################################
        if args.grad:
            # run singlepoint calculation
            result = calc.singlepoint(numbers, positions, chrg)

            timer.start("grad")
            (g,) = torch.autograd.grad(result.total.sum(), positions)
            timer.stop("grad")

            if args.verbosity >= 7:
                print_grad(g.clone(), numbers)
                print("")
        #####################################################

        if args.ir is True:
            # TODO: Better handling here
            positions.requires_grad_(True)
            calc.opts.scf.scf_mode = labels.SCF_MODE_FULL
            calc.opts.scf.mixer = labels.MIXER_ANDERSON

            timer.start("IR")
            ir_result = calc.ir(numbers, positions, chrg)
            ir_result.use_common_units()
            print("IR Frequencies\n", ir_result.freqs)
            print("IR Intensities\n", ir_result.ints)
            timer.stop("IR")
            calc.reset()

        if args.ir_numerical is True:
            timer.start("IR")
            ir_result = calc.ir_numerical(numbers, positions, chrg)
            ir_result.use_common_units()
            print("IR Frequencies\n", ir_result.freqs)
            print("IR Intensities\n", ir_result.ints)
            timer.stop("IR")

        if args.raman is True:
            # TODO: Better handling here
            positions.requires_grad_(True)
            calc.opts.scf.scf_mode = labels.SCF_MODE_FULL
            calc.opts.scf.mixer = labels.MIXER_ANDERSON

            # TODO: Better print handling
            timer.start("Raman")
            raman_result = calc.raman(numbers, positions, chrg)
            raman_result.use_common_units()
            print("Raman Frequencies\n", raman_result.freqs)
            print("Raman Intensities\n", raman_result.ints)
            timer.stop("Raman")

        if args.raman_numerical is True:
            timer.start("Raman Num")
            raman_result = calc.raman_numerical(numbers, positions, chrg)
            raman_result.use_common_units()
            print("Raman Frequencies\n", raman_result.freqs)
            print("Raman Intensities\n", raman_result.ints)
            timer.stop("Raman Num")

        if args.dipole is True:
            timer.start("Dipole")
            mu = calc.dipole(numbers, positions, chrg)
            timer.stop("Dipole")
            print("Dipole Moment\n", mu)

        if args.polarizability is True:
            timer.start("Polarizability")
            alpha = calc.polarizability(numbers, positions, chrg)
            timer.stop("Polarizability")
            print("Polarizability\n", alpha)

        if "energy" not in calc.cache:
            result = calc.singlepoint(numbers, positions, chrg)

            timer.print()
            result.print_energies()
            return result

    def __repr__(self) -> str:  # pragma: no cover
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.args})"
