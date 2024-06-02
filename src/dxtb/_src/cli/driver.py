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

from dxtb import Calculator
from dxtb._src import io
from dxtb._src.calculators.config import Config
from dxtb._src.calculators.result import Result
from dxtb._src.components.interactions.field import new_efield
from dxtb._src.constants import labels
from dxtb._src.timing import timer

__all__ = ["Driver"]


logger = logging.getLogger(__name__)

FILES = {"spin": ".UHF", "chrg": ".CHRG"}


def print_grad(grad, numbers) -> None:
    from tad_mctc.data import pse

    io.OutputHandler.write_stdout("\n\nForces")
    io.OutputHandler.write_stdout("------\n")

    # Iterate over the tensor and print
    io.OutputHandler.write_stdout(
        "  #  Z            dX              dY              dZ"
    )
    io.OutputHandler.write_stdout(
        "--------------------------------------------------------"
    )
    for i, row in enumerate(grad):
        # Get the atomic symbol corresponding to the atomic number
        symbol = pse.Z2S.get(int(numbers[i].item()), "?")
        io.OutputHandler.write_stdout(
            f"{i+1:>3}  {symbol:<2}  {row[0]:>15.9f} {row[1]:>15.9f} {row[2]:>15.9f}"
        )

    io.OutputHandler.write_stdout(
        "--------------------------------------------------------"
    )


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

    def singlepoint(self) -> Result | None:
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
        if args.json is True:
            io.OutputHandler.setup_json_logger()

        io.OutputHandler.header()
        io.OutputHandler.sysinfo()

        if args.detect_anomaly:
            # pylint: disable=import-outside-toplevel
            from torch.autograd.anomaly_mode import set_detect_anomaly

            set_detect_anomaly(True)

        dd = {"device": args.device, "dtype": args.dtype}

        # setup config and write to output
        config = Config.from_args(args)
        io.OutputHandler.write(config.info())

        # Broyden is not supported in full SCF mode
        if (
            config.scf.scf_mode == labels.SCF_MODE_FULL
            and config.scf.mixer == labels.MIXER_BROYDEN
        ):
            config.scf.mixer = labels.MIXER_ANDERSON

        # first tensor when using CUDA takes a long time to initialize...
        if "cuda" in str(dd["device"]):
            timer.start("Init GPU", parent_uid="Setup")
            _ = torch.tensor([0], **dd)
            timer.stop("Init GPU")
            del _

        timer.start("Read Files", parent_uid="Setup")

        if len(args.file) > 1:
            _n, _p = zip(
                *[read.read_from_path(f, ftype=args.filetype, **dd) for f in args.file]
            )
            numbers = pack(_n)
            positions = pack(_p)
        else:
            _n, _p = io.read_structure_from_file(args.file[0], args.filetype)
            numbers = torch.tensor(_n, dtype=torch.long, device=dd["device"])
            positions = torch.tensor(_p, **dd)

        timer.stop("Read Files")

        chrg = torch.tensor(self.chrg, **dd)

        if args.grad is True:
            positions.requires_grad = True

        if args.method.lower() == "gfn1" or args.method.lower() == "gfn1-xtb":
            # pylint: disable=import-outside-toplevel
            from dxtb import GFN1_XTB as par
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
            result = calc.singlepoint(positions, chrg)

            timer.start("grad")
            (g,) = torch.autograd.grad(result.total.sum(), positions)
            timer.stop("grad")

            timer.print()
            result.print_energies()

            if args.verbosity >= 7:
                print_grad(g.clone(), numbers)
                print("")

            io.OutputHandler.dump_warnings()
            return result
        #####################################################

        if args.forces is True:
            positions.requires_grad_(True)

            forces = calc.forces(positions, chrg)
            calc.reset()

            timer.print()
            print_grad(forces.clone(), numbers)
            # io.OutputHandler.dump_warnings()
            return

        if args.forces_numerical is True:
            timer.start("Forces")
            forces = calc.forces_numerical(positions, chrg)
            timer.stop("Forces")
            calc.reset()

            print_grad(forces.clone(), numbers)
            # io.OutputHandler.dump_warnings()
            return

        if args.hessian is True:
            positions.requires_grad_(True)

            timer.start("Hessian")
            hessian = calc.hessian(positions, chrg)
            timer.stop("Hessian")
            calc.reset()

            print(hessian.clone().detach())
            # io.OutputHandler.dump_warnings()
            return

        if args.hessian_numerical is True:
            positions.requires_grad_(True)

            timer.start("Hessian")
            hessian = calc.hessian_numerical(positions, chrg)
            timer.stop("Hessian")
            calc.reset()

            print(hessian.clone())
            # io.OutputHandler.dump_warnings()
            return

        if args.ir is True:
            # TODO: Better handling here
            positions.requires_grad_(True)
            calc.opts.scf.scf_mode = labels.SCF_MODE_FULL
            calc.opts.scf.mixer = labels.MIXER_ANDERSON

            timer.start("IR")
            ir_result = calc.ir(positions, chrg)
            ir_result.use_common_units()
            print("IR Frequencies\n", ir_result.freqs)
            print("IR Intensities\n", ir_result.ints)
            timer.stop("IR")
            calc.reset()

        if args.ir_numerical is True:
            timer.start("IR")
            ir_result = calc.ir_numerical(positions, chrg)
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
            raman_result = calc.raman(positions, chrg)
            raman_result.use_common_units()
            print("Raman Frequencies\n", raman_result.freqs)
            print("Raman Intensities\n", raman_result.ints)
            timer.stop("Raman")

        if args.raman_numerical is True:
            timer.start("Raman Num")
            raman_result = calc.raman_numerical(positions, chrg)
            raman_result.use_common_units()
            print("Raman Frequencies\n", raman_result.freqs)
            print("Raman Intensities\n", raman_result.ints)
            timer.stop("Raman Num")

        if args.dipole is True:
            calc.opts.scf.scf_mode = labels.SCF_MODE_FULL
            calc.opts.scf.mixer = labels.MIXER_ANDERSON

            timer.start("Dipole")
            mu = calc.dipole_analytical(positions, chrg)
            timer.stop("Dipole")
            print("Dipole Moment\n", mu)

        if args.polarizability is True:
            timer.start("Polarizability")
            alpha = calc.polarizability(positions, chrg)
            timer.stop("Polarizability")
            print("Polarizability\n", alpha)

        io.OutputHandler.dump_warnings()

        if "energy" not in calc.cache:
            result = calc.singlepoint(positions, chrg)

            timer.print()
            result.print_energies()
            io.OutputHandler.dump_warnings()
            return result

    def __repr__(self) -> str:  # pragma: no cover
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.args})"
