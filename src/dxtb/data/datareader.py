from __future__ import annotations

import os
from json import dump as json_dump_file
from json import dumps as json_dump
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..charges import ChargeModel, solve
from ..constants import units
from ..io import read
from ..ncoord import exp_count, get_coordination_number
from ..param import GFN1_XTB as par
from ..xtb import Calculator

FILES = {
    "charge": ".CHRG",
    "coord": "coord",
    "xyz": "mol.xyz",
    "gfn1": "gfn1.json",
    "gfn2": "gfn2.json",
    "uhf": ".UHF",
    "orca_engrad": "orca.engrad",
}


bh76rc = [
    "C2H5",
    "C2H6",
    "CH4",
    "H2",
    "H2O",
    "H2S",
    "HS",
    "NH",
    "NH2",
    "NH3",
    "O",
    "PH2",
    "PH3",
    "c2h4",
    "c2h5",
    "c3h7",
    "ch3",
    "ch3cl",
    "ch3f",
    "cl",
    "cl-",
    "clf",
    "co",
    "f",
    "f-",
    "f2",
    "fch3clcomp1",
    "fch3clcomp2",
    "h",
    "hcl",
    "hcn",
    "hco",
    "hf",
    "hn2",
    "hnc",
    "hoch3fcomp1",
    "hoch3fcomp2",
    "n2",
    "n2o",
    "oh",
]


def walklevel(some_dir: str | Path, level=1):
    """Identical to os.walk() but allowing for limited depth."""
    if isinstance(some_dir, Path):
        some_dir = str(some_dir)

    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), f"Not a directory: {some_dir}"
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


class Datareader:
    """Class to read in data from disk to data model."""

    def __init__(self, benchmark: str):

        path = str(Path(Path(__file__).resolve().parents[3], "data", benchmark))

        if not Path(path).is_dir():
            raise FileNotFoundError(f"Directory '{path}' not found.")

        self.benchmark = benchmark
        self.bpath = Path(path, "benchmark")
        self.path = path

    def get_sample_data(self):
        """Fetch all data given in root directory."""
        self.data = []
        self.file_list = []

        # loop through folders + subfolders only
        for (dirpath, _, filenames) in walklevel(self.bpath, level=2):
            if FILES["coord"] not in filenames and FILES["xyz"] not in filenames:
                continue

            egfn1, ggfn1 = [], []
            if FILES["gfn1"] in filenames:
                egfn1, ggfn1 = read.read_tblite_gfn("/".join([dirpath, FILES["gfn1"]]))

            egfn2, ggfn2 = [], []
            if FILES["gfn2"] in filenames:
                egfn2, ggfn2 = read.read_tblite_gfn("/".join([dirpath, FILES["gfn2"]]))

            eref, gref = 0.0, []
            if FILES["orca_engrad"] in filenames:
                eref, gref = read.read_orca_engrad(
                    "/".join([dirpath, FILES["orca_engrad"]])
                )

            # read coord/xyz file
            if FILES["coord"] in filenames:
                geo = read.read_coord("/".join([dirpath, FILES["coord"]]))
            elif FILES["xyz"] in filenames:
                geo = read.read_xyz("/".join([dirpath, FILES["xyz"]]))
            else:
                raise FileNotFoundError(f"No coord/xyz file found in '{dirpath}'.")

            assert len(geo[0]) == 4
            positions = [g[:3] for g in geo]
            numbers = [g[-1] for g in geo]

            # read chrg file
            if FILES["charge"] in filenames:
                chrg = read.read_chrg("/".join([dirpath, FILES["charge"]]))
            else:
                chrg = 0

            # read uhf file
            if FILES["uhf"] in filenames:
                uhf = read.read_chrg("/".join([dirpath, FILES["uhf"]]))
            else:
                uhf = 0

            self.data.append(
                [
                    numbers,
                    positions,
                    chrg,
                    uhf,
                    egfn1,
                    ggfn1,
                    egfn2,
                    ggfn2,
                    eref,
                    gref,
                ]
            )

            sample = dirpath.replace(self.path, "").replace(
                "benchmark/", f"{self.benchmark}:"
            )

            if sample.startswith("/"):
                sample = sample[1:]
            self.file_list.append(sample)

            # create entries for BH76RC separately (creates duplicates)
            if self.benchmark == "GMTKN55":
                if "BH76/" in sample:
                    molecule = sample.rsplit("/", 1)[1]

                    if molecule in bh76rc:
                        self.data.append(
                            [
                                numbers,
                                positions,
                                chrg,
                                uhf,
                                egfn1,
                                ggfn1,
                                egfn2,
                                ggfn2,
                                eref,
                                gref,
                            ]
                        )
                        self.file_list.append(f"{self.benchmark}:BH76RC/{molecule}")

        if self.benchmark == "GMTKN55":
            # GMTKN55 plus duplicates for BH76RC
            assert len(self.file_list) == 2462 + 40

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({len(self.file_list)} files)"

    def slice(self, idx: int | slice) -> None:
        if isinstance(idx, slice):
            self.data = self.data[idx]
            self.file_list = self.file_list[idx]
        elif isinstance(idx, int):
            self.data = [self.data[idx]]
            self.file_list = [self.file_list[idx]]
        else:
            raise TypeError(f"Invalid index '{idx}' type.")

    def get_by_name(
        self, name: str, inplace: bool = True
    ) -> tuple[list[str], list[list]]:
        """
        Get data by name.

        Parameters
        ----------
        name : str
            Search string for file name.
        inplace : bool, optional
            If `True`, modify `self.data` and `self.file_list`. Defaults to `True`.

        Returns
        -------
        tuple[list[str], list[list]]
            Tuple of file names and data.

        Examples
        --------
        Get only data for aluminum.
        >>> name = "AL"
        >>> data = Datareader("PTB")
        >>> data.get_sample_data()
        >>> data.get_by_name(f"/{name}")
        >>> data.sort()
        >>> data.create_sample_json(f"samples_{name}.json")
        """
        zipped_lists = zip(self.file_list, self.data)

        selected_pairs = [x for x in zipped_lists if name in x[0]]
        if len(selected_pairs) == 0:
            raise ValueError(f"No data found for name '{name}'.")

        tuples = zip(*selected_pairs)
        file_list, data = (list(tuple) for tuple in tuples)

        if inplace is True:
            self.data = data
            self.file_list = file_list

        return file_list, data

    def sort(self) -> None:
        """Sort data by file name (case-insensitive)."""
        zipped_lists = zip(self.file_list, self.data)
        sorted_pairs = sorted(zipped_lists, key=lambda s: s[0].casefold())
        tuples = zip(*sorted_pairs)
        file_list, data = (list(tuple) for tuple in tuples)

        self.data = data
        self.file_list = file_list

    def create_sample_json(
        self,
        out_name: str = "samples.json",
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Create the samples.json file containing the features for each sample.

        Parameters
        ----------
        out_name : str, optional
            Name of json file, by default "samples.json"
        dtype : torch.dtype, optional
            Dtype of float values. Defaults to `None`.
        device : torch.device, optional
            Device on which the tensors reside. Defaults to `None`.

        Raises
        ------
        FileNotFoundError
            Parent directory of `out_name` does not exist.

        Example
        -------
        >>> from xtbml.data.datareader import Datareader
        >>> data = Datareader()
        >>> data.sort()
        >>> data.slice(slice(600))
        >>> data.create_sample_json()
        """

        def calc_singlepoint(numbers, positions, charge, unpaired_e):
            """Calculate QM features based on classicals input, such as geometry."""

            # setup calculator
            calc = Calculator(numbers, positions, par)
            result = calc.singlepoint(numbers, positions, charge, {"verbosity": 0})

            # partial charges
            cn = get_coordination_number(numbers, positions, exp_count)
            eeq = ChargeModel.param2019()
            _, qat = solve(numbers, positions, charge, eeq, cn)

            return (
                result.hcore,
                result.overlap,
                cn,
                result.repulsion,
                result.dispersion,
                qat,
            )

        path = Path(self.path, out_name)

        # opening curly bracket for json
        with open(path, "w", encoding="utf-8") as f:
            f.write("{")

        # append each sample to json
        for i, (file, data) in enumerate(zip(self.file_list, self.data)):
            print(i, file)
            with open(path, "a", encoding="utf-8") as f:
                d = {}

                numbers = torch.tensor(data[0], device=device, dtype=torch.long)
                positions = torch.tensor(data[1], device=device, dtype=dtype)
                charge = torch.tensor(data[2], device=device, dtype=torch.int16)
                unpaired_e = torch.tensor(data[3], device=device, dtype=torch.int16)

                h, o, cnum, erep, edisp, qat = calc_singlepoint(
                    numbers, positions, charge, unpaired_e
                )

                d[file] = dict(
                    numbers=data[0],
                    positions=data[1],
                    unpaired_e=data[3],
                    charges=data[2],
                    egfn1=[i * units.AU2KCAL for i in data[4]],
                    ggfn1=data[5],
                    egfn2=[i * units.AU2KCAL for i in data[6]],
                    ggfn2=data[7],
                    eref=units.AU2KCAL * data[8],
                    gref=data[9],
                    edisp=[i * units.AU2KCAL for i in edisp.tolist()],
                    erep=[i * units.AU2KCAL for i in erep.tolist()],
                    qat=qat.tolist(),
                    cn=cnum.tolist(),
                    h0=h.tolist(),
                    ovlp=o.tolist(),
                )

                out = json_dump(d)

                # remove curly braces to basically convert it to an entry
                out = out[1:-1]
                f.write(out)

                # do not add comma for last item
                if file != self.file_list[-1]:
                    f.write(",")

        # closing curly bracket for json
        with open(path, "a", encoding="utf-8") as f:
            f.write("}")

    def create_reaction_json(self, out_name: str = "reactions.json") -> None:
        """Create the json

        Parameters
        ----------
        out_name : str, optional
            Name of json file, by default "reactions.json"

        Raises
        ------
        FileNotFoundError
            Energy or ".ref" file does not exist.

        Example
        -------
        >>> from xtbml.data.datareader import Datareader
        >>> data = Datareader("MOR41")
        >>> data.create_reaction_json()
        """

        if self.benchmark == "GMTKN55":
            raise NotImplementedError(
                "Nested directory structure cannot be handled currently."
            )

        def get_energy(system, energy_file="energy"):
            path = Path(system, energy_file)
            if not path.is_file():
                raise FileNotFoundError(f"File '{path}' not found.")

            energy, _ = read.read_tblite_gfn(path)

            if energy == 0 or not energy:
                raise ValueError(f"No energy found in {path}.")

            return energy

        reaction = {}
        count = 1

        path = Path(self.bpath, ".res")
        if not path.is_file():
            raise FileNotFoundError(f"File '{path}' not found.")

        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("$tmer") or line.startswith("tmer2++"):
                    print(f"{self.benchmark}_{count}", end="\r", flush=True)
                    line = line.strip()
                    line = line.split("#")[0]
                    line = line.replace("$tmer ", "")
                    line = line.replace("tmer2++ ", "")

                    mols = line.split(" x ")[0].split()
                    mols = [m.replace("/$f", "") for m in mols]
                    stoichs = line.split(" x ")[1].split("$w")[0].split()
                    stoichs = [int(s) for s in stoichs]
                    ref = float(line.split()[-1])

                    e_gfn1 = 0
                    e_gfn2 = 0
                    systems = []
                    nus = []
                    for mol, stoich in zip(mols, stoichs):
                        mol_path = Path(self.bpath, mol)
                        energy1 = get_energy(mol_path, "gfn1.json")
                        energy2 = get_energy(mol_path, "gfn2.json")

                        e_gfn1 += stoich * energy1
                        e_gfn2 += stoich * energy2

                        systems.append(mol)
                        nus.append(stoich)

                    reaction[f"{self.benchmark}_{count}"] = dict(
                        nu=nus,
                        partners=systems,
                        egfn1=e_gfn1 * units.AU2KCAL,
                        egfn2=e_gfn2 * units.AU2KCAL,
                        eref=ref,
                    )

                    count += 1

        with open(Path(self.bpath.parents[0], out_name), "w", encoding="utf-8") as f:
            json_dump_file(reaction, f)

    @staticmethod
    def get_dataloader(geometry, cfg: dict | None = None) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over Geometry object.

        Parameters
        ----------
        geometry : Tensor
            Geometry object containing _all_ geometries in dataset.
        cfg : dict | None, optional
            Optional configuration for dataloader settings. Defaults to None.

        Returns
        -------
        DataLoader
            Pytorch dataloader
        """

        def collate_fn(batch):
            """Custom collate fn for loading Geometry objects in batch form."""
            # add geometries together
            batch_geometry = batch[0]
            for i in range(1, len(batch)):
                batch_geometry = batch_geometry + batch[i]
            return batch_geometry

        if cfg is None:
            cfg = {"collate_fn": collate_fn}
        elif "collate_fn" not in cfg:
            cfg["collate_fn"] = collate_fn

        return DataLoader(geometry, **cfg)
