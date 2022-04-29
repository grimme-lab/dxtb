import os
from typing import List, Tuple, Optional
import torch
from torch.utils.data import DataLoader

from ..exlibs.tbmalt import Geometry
from ..data.covrad import to_number
from ..constants import FLOAT64


def walklevel(some_dir: str, level=1):
    """Identical to os.walk() but allowing for limited depth."""
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), f"Not a directory: {some_dir}"
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def read_coord(
    fp: str,
    breakpoints=["$user-defined bonds", "$redundant", "$end"],
    dtype: Optional[torch.dtype] = FLOAT64,
):
    """Read a coord file."""
    arr = []
    with open(fp, "r") as file:
        lines = file.readlines()
        for line in lines:
            l = line.split()
            # skip
            if len(l) == 0:
                continue
            elif any([bp in line for bp in breakpoints]):
                break
            elif l[0].startswith("$"):
                continue
            try:
                x, y, z = float(l[0]), float(l[1]), float(l[2])
                atm = to_number(l[3])
                arr.append([x, y, z, atm])
            except ValueError as e:
                print(e)
                print(f"WARNING: No correct values. Skip sample {fp}")
                return
    return torch.tensor(arr, dtype=dtype)


def read_chrg(fp: str):
    """Read a chrg (or uhf) file."""
    with open(fp, "r") as file:
        chrg = int(file.read())
    return chrg


def read_energy(fp: str):
    """Read energy file (in TM format)."""
    with open(fp, "r") as file:
        for i, line in enumerate(file):
            # energy is three times on second line (TM format)
            if i == 1:
                return float(line.strip().split()[-1])


class Datareader:
    """Class to read in data from disk to Geometry object."""

    def fetch_data(root: str) -> Tuple[List, List]:
        """Fetch all data given in root directory."""

        FILE_CHARGE = ".CHRG"
        FILE_COORD = "coord"
        FILE_ENERGY = "energy"
        FILE_UHF = ".UHF"

        data = []
        file_list = []

        # loop through folders + subfolders only
        for (dirpath, dirnames, filenames) in walklevel(root, level=2):
            if FILE_COORD not in filenames:
                continue

            # read coord file
            geo = read_coord("/".join([dirpath, FILE_COORD]))
            assert len(geo.shape) == 2
            xyz = geo[:, :3].tolist()
            q = list(map(int, geo[:, -1].tolist()))

            # read chrg file
            if FILE_CHARGE in filenames:
                chrg = read_chrg("/".join([dirpath, FILE_CHARGE]))
            else:
                chrg = 0

            # read uhf file
            if FILE_UHF in filenames:
                uhf = read_chrg("/".join([dirpath, FILE_UHF]))
            else:
                uhf = 0

            # read energy file
            if FILE_ENERGY in filenames:
                energy = read_energy("/".join([dirpath, FILE_ENERGY]))
            else:
                energy = 0

            data.append([xyz, q, chrg, uhf, energy])
            file_list.append(dirpath.replace(root, ""))

        return data, file_list

    def setup_geometry(
        data: torch.Tensor,
        dtype: Optional[torch.dtype] = FLOAT64,
        dtype_int: Optional[torch.dtype] = torch.long,  # TODO: adapt default dtype
        device: Optional[torch.device] = None,
    ) -> Geometry:
        """Convert data tensor into batched geometry object."""

        # TODO: add default dtype and device depending on config

        positions = [torch.tensor(xyz, device=device, dtype=dtype) for xyz, *_ in data]
        atomic_numbers = [
            torch.tensor(q, device=device, dtype=dtype_int) for _, q, *_ in data
        ]

        charges = [
            torch.tensor(chrg, device=device, dtype=dtype_int)
            for *_, chrg, _, _ in data
        ]
        unpaired_e = [
            torch.tensor(uhf, device=device, dtype=dtype_int) for *_, uhf, _ in data
        ]

        # convert into geometry object
        return Geometry(atomic_numbers, positions, charges, unpaired_e, units="bohr")

    def get_dataloader(geometry: Geometry, cfg: Optional[dict] = None) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over Geometry object.

        Args:
            geometry (Geometry): Geometry object containing _all_ geometries in dataset.
            cfg (dict, optional): Optional configuration for dataloader settings. Defaults to None.

        Returns:
            DataLoader: Pytorch dataloader
        """

        def collate_fn(batch):
            """Custom collate fn for loading Geometry objects in batch form."""
            # add geometries together
            batch_geometry = batch[0]
            for i in range(1, len(batch)):
                batch_geometry = batch_geometry + batch[i]
            return batch_geometry

        if "collate_fn" not in cfg:
            cfg["collate_fn"] = collate_fn

        return DataLoader(geometry, **cfg)
