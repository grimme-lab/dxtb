import os
from typing import List, Tuple
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


def read_coord(fp: str, breakpoints=["$user-defined bonds", "$redundant", "$end"]):
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
    return torch.tensor(arr, dtype=FLOAT64)


def read_chrg(fp: str):
    """Read a chrg (or uhf) file."""
    with open(fp, "r") as file:
        chrg = int(file.read())
    return chrg


class Datareader:
    """Class to read in data from disk to Geometry object."""

    def fetch_data(root: str) -> Tuple[List, List]:
        """Fetch all data given in root directory."""

        data = []
        file_list = []

        # loop through folders + subfolders only
        for (dirpath, dirnames, filenames) in walklevel(root, level=2):
            if "coord" not in filenames:
                continue

            # read coord file
            geo = read_coord("/".join([dirpath, "coord"]))
            assert len(geo.shape) == 2
            xyz = geo[:, :3].tolist()
            q = list(map(int, geo[:, -1].tolist()))

            # read chrg file
            if ".CHRG" in filenames:
                chrg = read_chrg("/".join([dirpath, ".CHRG"]))
            else:
                chrg = 0

            # read uhf file
            if ".UHF" in filenames:
                uhf = read_chrg("/".join([dirpath, ".UHF"]))
            else:
                uhf = 0

            if len(filenames) > 3:
                print(dirpath, filenames)
                raise IOError

            data.append([xyz, q, chrg, uhf])
            file_list.append(dirpath)

        return data, file_list

    def setup_geometry(
        data: torch.Tensor, dtype: torch.dtype = FLOAT64, device=None
    ) -> Geometry:
        """Convert data tensor into batched geometry object."""

        # TODO: add default dtype and device depending on config

        positions = [torch.tensor(xyz, device=device, dtype=dtype) for xyz, *_ in data]
        atomic_numbers = [torch.tensor(q, device=device) for _, q, *_ in data]

        charges = [torch.tensor(chrg, device=device) for *_, chrg, _ in data]
        unpaired_e = [torch.tensor(uhf, device=device) for *_, uhf in data]

        # convert into geometry object
        return Geometry(atomic_numbers, positions, charges, unpaired_e, units="bohr")

    def get_dataloader(geometry: Geometry, cfg: dict = None) -> DataLoader:
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
