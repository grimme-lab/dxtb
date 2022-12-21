"""
IO utility for reading files.
"""

from json import loads as json_load
from pathlib import Path

import torch

from .._types import PathLike
from ..constants import ATOMIC_NUMBER, units


def check_xyz(fp: PathLike, xyz: list[list[float]]) -> list[list[float]]:
    """
    Check coordinates of file. Particularly, we check for the last coordinate
    being at the origin as this might clash with padding.

    Parameters
    ----------
    fp : PathLike
        Path to coordinate file.
    xyz : list[list[float]]
        Coordinates of structure.

    Returns
    -------
    list[list[float]]
        Coordinates of structure.

    Raises
    ------
    ValueError
        File is actually empty or last coordinate is at origin.
    """

    if len(xyz) == 0:
        raise ValueError(f"File '{fp}' is empty.")
    elif len(xyz) == 1:
        if xyz[-1] == [0.0, 0.0, 0.0]:
            xyz[-1][0] = 1.0
    else:
        if xyz[-1] == [0.0, 0.0, 0.0]:
            raise ValueError(
                f"Last coordinate is zero in '{fp}'. This will clash with padding."
            )

    return xyz


def read_structure_from_file(
    file: PathLike, ftype: str | None = None
) -> tuple[list[int], list[list[float]]]:
    """
    Helper to read the structure from the given file.

    Parameters
    ----------
    file : PathLike
        Path of file containing the structure.
    ftype : str | None, optional
        File type. Defaults to `None`, i.e., infered from the extension.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Lists of atoms and coordinates.

    Raises
    ------
    FileNotFoundError
        File given does not exist.
    NotImplementedError
        Reader for specific file type not implemented.
    ValueError
        Unknown file type.
    """

    f = Path(file)
    if f.exists() is False:
        raise FileNotFoundError(f"File '{f}' not found.")

    if ftype is None:
        ftype = f.suffix.lower()[1:]
    fname = f.name.lower()

    if ftype in ("xyz", "log"):
        numbers, positions = read_xyz(f)
    elif ftype in ("tmol", "tm", "turbomole") or fname in ("coord"):
        numbers, positions = read_coord(f)
    elif ftype in ("mol", "sdf", "gen", "pdb"):
        raise NotImplementedError(
            f"Filetype '{ftype}' recognized but no reader available."
        )
    elif ftype in ("qchem"):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Q-Chem) recognized but no reader available."
        )
    elif ftype in ("poscar", "contcar", "vasp", "crystal") or fname in (
        "poscar",
        "contcar",
        "vasp",
    ):
        raise NotImplementedError(
            "VASP/CRYSTAL file recognized but no reader available."
        )
    elif ftype in ("ein", "gaussian"):
        raise NotImplementedError(
            f"Filetype '{ftype}' (Gaussian) recognized but no reader available."
        )
    elif ftype in ("json", "qcschema"):
        numbers, positions = read_qcschema(f)
    else:
        raise ValueError(f"Unknown filetype '{ftype}' in '{f}'.")

    return numbers, positions


def read_xyz(fp: PathLike) -> tuple[list[int], list[list[float]]]:
    """
    Read xyz file.

    Parameters
    ----------
    fp : PathLike
        Path to coordinate file.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Lists containing the atomic numbers and coordinates.
    """
    atoms = []
    xyz = []
    num_atoms = 0

    with open(fp, encoding="utf-8") as file:
        for line_number, line in enumerate(file):
            if line_number == 0:
                num_atoms = int(line)
            elif line_number == 1:
                pass
            else:
                l = line.strip().split()
                atom, x, y, z = l
                xyz.append([i * units.AA2AU for i in [float(x), float(y), float(z)]])
                atoms.append(ATOMIC_NUMBER[atom.title()])

    if len(xyz) != num_atoms:
        raise ValueError(f"Number of atoms in {fp} does not match.")

    xyz = check_xyz(fp, xyz)
    return atoms, xyz


def read_qcschema(fp: PathLike) -> tuple[list[int], list[list[float]]]:
    """
    Read json/QCSchema file.

    Parameters
    ----------
    fp : PathLike
        Path to coord file.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Lists containing the atomic numbers and coordinates.
    """

    with open(fp, encoding="utf-8") as file:
        data = json_load(file.read())

    if "molecule" not in data:
        raise KeyError(f"Invalid schema: Key 'molecule' not found in '{fp}'.")

    mol = data["molecule"]

    if "symbols" not in mol:
        raise KeyError(f"Invalid schema: Key 'symbols' not found in '{fp}'.")
    if "geometry" not in mol:
        raise KeyError(f"Invalid schema: Key 'geometry' not found in '{fp}'.")

    atoms = []
    for atom in mol["symbols"]:
        atoms.append(ATOMIC_NUMBER[atom.title()])

    xyz = []
    geo = mol["geometry"]
    for i in range(0, len(geo), 3):
        xyz.append([float(geo[i]), float(geo[i + 1]), float(geo[i + 2])])

    xyz = check_xyz(fp, xyz)
    return atoms, xyz


def read_coord(fp: PathLike) -> tuple[list[int], list[list[float]]]:
    """
    Read Turbomole/coord file.

    Parameters
    ----------
    fp : PathLike
        Path to coord file.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        Lists containing the atomic numbers and coordinates.
    """
    atoms = []
    xyz = []
    breakpoints = ["$user-defined bonds", "$redundant", "$end", "$periodic"]

    with open(fp, encoding="utf-8") as file:
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
                x, y, z, atom = l
                xyz.append([float(x), float(y), float(z)])
                atoms.append(ATOMIC_NUMBER[atom.title()])
            except ValueError as e:
                print(e)
                print(f"WARNING: No correct values. Skip sample {fp}")

    xyz = check_xyz(fp, xyz)
    return atoms, xyz


def read_chrg(fp: PathLike) -> int:
    """Read a chrg (or uhf) file."""

    if not Path(fp).is_file():
        return 0

    with open(fp, encoding="utf-8") as file:
        return int(file.read())


def read_energy(fp: str) -> float:
    """Read energy file in TM format (energy is three times on second line)."""
    with open(fp, encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i == 1:
                return float(line.strip().split()[-1])

        raise ValueError(f"File '{fp}' is not in Turbomole format.")


def read_tblite_gfn(fp: Path | str) -> tuple[float, list[float]]:
    """Read energy file from tblite json output."""
    with open(fp, encoding="utf-8") as file:
        data = json_load(file.read())

        return data["energies"], torch.tensor(data["gradient"]).reshape(-1, 3).tolist()


def read_orca_engrad(fp: Path | str) -> tuple[float, list[float]]:
    """Read ORCA's engrad file."""
    start_grad = -1
    grad = []

    start_energy = -1
    energy = 0.0
    with open(fp, encoding="utf-8") as file:
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
