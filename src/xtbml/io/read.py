from pathlib import Path
from json import dumps as json_dump
from json import dump as json_dump_file
from json import loads as json_load
import torch

from ..constants import ATOMIC_NUMBER
from ..typing import Literal


def read_geo(fp, frmt: Literal["xyz", "coord"] = "xyz") -> list[list[float | int]]:
    """Read geometry file.
    Parameters
    ----------
    fp : str | Path
        Path to coord file.
    frmt : str
        Format of the file.
    Returns
    -------
    list[list[float | int]]
        list containing the atomic numbers and coordinates.
    """
    if frmt == "xyz":
        return read_xyz(fp)
    elif frmt == "coord":
        return read_coord(fp)
    else:
        raise ValueError(f"Unknown format: {frmt}")


def read_xyz(fp: str | Path) -> list[list[float | int]]:
    """
    Read xyz file.

    Parameters
    ----------
    fp : str | Path
        Path to coord file.

    Returns
    -------
    list[list[float | int]]
        list containing the atomic numbers and coordinates.
    """
    arr = []
    num_atoms = 0

    with open(fp, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                pass
            else:
                l = line.strip().split()
                x, y, z = float(l[1]), float(l[2]), float(l[3])
                atm = ATOMIC_NUMBER[l[0].title()]
                arr.append([x, y, z, atm])

    if len(arr) == 1:
        if arr[0][0] == arr[0][1] == arr[0][2] == 0:
            arr[0][0] = 1.0

    if len(arr) != num_atoms:
        raise ValueError(f"Number of atoms in {fp} does not match.")

    return arr


def read_coord(fp: str | Path) -> list[list[float | int]]:
    """
    Read coord file.

    Parameters
    ----------
    fp : str | Path
        Path to coord file.

    Returns
    -------
    list[list[float | int]]
        list containing the atomic numbers and coordinates.
    """
    arr = []
    breakpoints = ["$user-defined bonds", "$redundant", "$end", "$periodic"]

    with open(fp, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()

            # skip
            if len(l) == 0:
                continue
            elif any(bp in line for bp in breakpoints):
                break
            elif l[0].startswith("$"):
                continue
            try:
                x, y, z = float(l[0]), float(l[1]), float(l[2])
                atm = ATOMIC_NUMBER[l[3].title()]
                arr.append([x, y, z, atm])
            except ValueError as e:
                print(e)
                print(f"WARNING: No correct values. Skip sample {fp}")

    # make sure last coord is not 0,0,0
    if len(arr) == 0:
        raise ValueError(f"File {fp} is empty.")
    elif len(arr) == 1:
        if arr[-1][:3] == [0.0, 0.0, 0.0]:
            arr[-1][0] = 1.0
    else:
        if arr[-1][:3] == [0.0, 0.0, 0.0]:
            raise ValueError(
                f"Last coordinate is zero in '{fp}'. This will clash with padding."
            )

    return arr


def read_chrg(fp: str | Path) -> int:
    """Read a chrg (or uhf) file."""

    if not Path(fp).is_file():
        return 0

    with open(fp, "r", encoding="utf-8") as file:
        return int(file.read())


def read_energy(fp: str) -> float:
    """Read energy file in TM format (energy is three times on second line)."""
    with open(fp, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == 1:
                return float(line.strip().split()[-1])

        raise ValueError(f"File '{fp}' is not in Turbomole format.")


def read_tblite_gfn(fp: Path | str) -> tuple[float, list[float]]:
    """Read energy file from tblite json output."""
    with open(fp, "r", encoding="utf-8") as file:
        data = json_load(file.read())

        return data["energies"], torch.tensor(data["gradient"]).reshape(-1, 3).tolist()


def read_orca_engrad(fp: Path | str) -> tuple[float, list[float]]:
    """Read ORCA's engrad file."""
    start_grad = -1
    grad = []

    start_energy = -1
    energy = 0.0
    with open(fp, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            # energy
            if line.startswith("# The current total energy in Eh"):
                start_energy = i + 2

            if i == start_energy:
                energy = float(line.strip())
                start_energy = -1

            # gradient
            if line.startswith("# The current gradient in Eh/bohr"):
                start_grad = i + 2

            if i == start_grad:
                # abort if we hit the next "#"
                if line.startswith("#"):
                    break

                grad.append(float(line.strip()))
                start_grad += 1

    return energy, torch.tensor(grad).reshape(-1, 3).tolist()
