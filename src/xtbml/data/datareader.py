from __future__ import annotations
from json import dumps as json_dump
from json import dump as json_dump_file
from json import loads as json_load
import os
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import DataLoader

from xtbml.constants.torch import FLOAT32, FLOAT64
from xtbml.constants.units import AU2KCAL
from xtbml.data.covrad import to_number


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


def read_geo(fp, format: Literal["xyz", "coord"] = "xyz") -> list[list[float | int]]:
    """Read geometry file.

    Parameters
    ----------
    fp : str | Path
        Path to coord file.
    format : str
        Format of the file.

    Returns
    -------
    list[list[float | int]]
        list containing the atomic numbers and coordinates.
    """
    if format == "xyz":
        return read_xyz(fp)
    elif format == "coord":
        return read_coord(fp)
    else:
        raise ValueError(f"Unknown format: {format}")


def read_xyz(fp: str | Path) -> list[list[float | int]]:
    """Read xyz file.

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
                atm = to_number(l[0])
                arr.append([x, y, z, atm])

    if len(arr) == 1:
        if arr[0][0] == arr[0][1] == arr[0][2] == 0:
            arr[0][0] = 1.0

    if len(arr) != num_atoms:
        raise ValueError(f"Number of atoms in {fp} does not match.")

    return arr


def read_coord(fp: str | Path) -> list[list[float | int]]:
    """Read coord file.

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
                atm = to_number(l[3])
                arr.append([x, y, z, atm])
            except ValueError as e:
                print(e)
                print(f"WARNING: No correct values. Skip sample {fp}")

    if len(arr) == 1:
        if arr[0][0] == arr[0][1] == arr[0][2] == 0:
            arr[0][0] = 1.0

    return arr


def read_chrg(fp: str) -> int:
    """Read a chrg (or uhf) file."""
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


class Datareader:
    """Class to read in data from disk to Geometry object."""

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

        FILE_CHARGE = ".CHRG"
        FILE_COORD = "coord"
        FILE_XYZ = "mol.xyz"
        FILE_GFN1 = "gfn1.json"
        FILE_GFN2 = "gfn2.json"
        FILE_UHF = ".UHF"
        FILE_ORCA_ENGRAD = "orca.engrad"

        # loop through folders + subfolders only
        for (dirpath, _, filenames) in walklevel(self.bpath, level=2):
            if FILE_COORD not in filenames and FILE_XYZ not in filenames:
                continue

            gfn1_energy, gfn1_grad = [], []
            if FILE_GFN1 in filenames:
                gfn1_energy, gfn1_grad = read_tblite_gfn("/".join([dirpath, FILE_GFN1]))

            gfn2_energy, gfn2_grad = [], []
            if FILE_GFN2 in filenames:
                gfn2_energy, gfn2_grad = read_tblite_gfn("/".join([dirpath, FILE_GFN2]))

            dft_energy, dft_grad = [], []
            if FILE_ORCA_ENGRAD in filenames:
                dft_energy, dft_grad = read_orca_engrad(
                    "/".join([dirpath, FILE_ORCA_ENGRAD])
                )

            # read coord/xyz file
            if FILE_COORD in filenames:
                geo = read_geo("/".join([dirpath, FILE_COORD]), format="coord")
            elif FILE_XYZ in filenames:
                geo = read_geo("/".join([dirpath, FILE_XYZ]), format="xyz")
            else:
                raise FileNotFoundError(f"No coord/xyz file found in '{dirpath}'.")

            assert len(geo[0]) == 4
            xyz = [g[:3] for g in geo]
            q = [g[-1] for g in geo]

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

            self.data.append(
                [
                    xyz,
                    q,
                    chrg,
                    uhf,
                    gfn1_energy,
                    gfn1_grad,
                    gfn2_energy,
                    gfn2_grad,
                    dft_energy,
                    dft_grad,
                ]
            )

            sample = dirpath.replace(self.path, "").replace("benchmark", self.benchmark)
            if sample.startswith("/"):
                sample = sample[1:]
            self.file_list.append(sample)

            # create entries for BH76RC separately (creates duplicates)
            if self.benchmark == "GMTKN55":
                if "BH76/" in sample:
                    molecule = sample.rsplit("/", 1)[1]

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

                    if molecule in bh76rc:
                        self.data.append(
                            [
                                xyz,
                                q,
                                chrg,
                                uhf,
                                gfn1_energy,
                                gfn1_energy_atom_resolved,
                                gfn2_energy,
                                gfn2_energy_atom_resolved,
                            ]
                        )
                        self.file_list.append(f"BH76RC/{molecule}")

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
            If True, modify self.data and self.file_list.

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
        file_list, data = [list(tuple) for tuple in tuples]

        if inplace is True:
            self.data = data
            self.file_list = file_list

        return file_list, data

    def sort(self) -> None:
        """Sort data by file name (case-insensitive)."""
        zipped_lists = zip(self.file_list, self.data)
        sorted_pairs = sorted(zipped_lists, key=lambda s: s[0].casefold())
        tuples = zip(*sorted_pairs)
        file_list, data = [list(tuple) for tuple in tuples]

        self.data = data
        self.file_list = file_list

    # FIXME: Currently not in use
    def get_geometry_data(
        self,
        dtype: Optional[torch.dtype] = FLOAT64,
        dtype_int: Optional[torch.dtype] = torch.long,  # TODO: adapt default dtype
        device: Optional[torch.device] = None,
    ) -> tuple:
        """Convert data tensor into batched geometry object."""

        # TODO: add default dtype and device depending on config

        positions = [
            torch.tensor(xyz, device=device, dtype=dtype) for xyz, *_ in self.data
        ]
        atomic_numbers = [
            torch.tensor(q, device=device, dtype=torch.long) for _, q, *_ in self.data
        ]

        charges = [
            torch.tensor(chrg, device=device, dtype=dtype_int)
            for *_, chrg, _, _ in self.data
        ]
        unpaired_e = [
            torch.tensor(uhf, device=device, dtype=dtype_int)
            for *_, uhf, _ in self.data
        ]

        # convert into geometry object
        return atomic_numbers, positions, charges, unpaired_e

    # FIXME: Dependency on Geometry object from tbmalt through Calculator/Hamiltonian
    def create_sample_json(
        self,
        out_name: str = "samples.json",
        dtype: Optional[torch.dtype] = FLOAT32,
        device: Optional[torch.device] = None,
    ) -> None:
        """Create the samples.json file containing the features for each sample.

        Parameters
        ----------
        out_name : str, optional
            Name of json file, by default "samples.json"
        dtype : Optional[torch.dtype], optional
            Dtype of float values, by default FLOAT32
        device : Optional[torch.device], optional
            Device on which the tensors reside, by default None

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

        # pylint: disable=import-outside-toplevel
        import tad_dftd3 as d3
        from xtbml import charges
        from xtbml.adjlist import AdjacencyList
        from xtbml.basis.type import get_cutoff
        from xtbml.repulsion.repulsion import RepulsionFactory
        from xtbml.xtb.calculator import Calculator
        from xtbml.ncoord.ncoord import get_coordination_number, exp_count
        from xtbml.param.gfn1 import GFN1_XTB as gfn1_par
        from xtbml.exlibs.tbmalt import Geometry
        from xtbml.exlibs.tbmalt.batch import deflate

        def calc_singlepoint(numbers, positions, charge, unpaired_e):
            """Calculate QM features based on classicals input, such as geometry."""

            # CALC FEATURES

            # NOTE: singlepoint calculation so far only working with no padding
            # quick and dirty hack to get rid of padding
            mol = Geometry(
                atomic_numbers=deflate(numbers),
                positions=deflate(positions, axis=1),
                charges=charge,
                unpaired_e=unpaired_e,
            )

            # check underlying data consistent
            assert (
                numbers.storage().data_ptr() == mol.atomic_numbers.storage().data_ptr()
            )

            # setup calculator
            par = gfn1_par
            calc = Calculator(mol, par)

            # prepate cutoffs and lattice
            # trans = get_lattice_points(mol_periodic, mol_lattice, cn_cutoff)
            cn = get_coordination_number(numbers, positions, exp_count)

            cutoff = get_cutoff(calc.basis)
            # trans = get_lattice_points(mol_periodic, mol_lattice, cutoff)
            trans = torch.zeros([3, 1])
            adj = AdjacencyList(mol, trans, cutoff)

            # build hamiltonian
            h, overlap_int = calc.hamiltonian.build(calc.basis, adj, cn)

            # repulsion
            repulsion = RepulsionFactory(
                numbers=mol.atomic_numbers,
                positions=mol.positions,
                req_grad=False,
                cutoff=torch.tensor(cutoff),
            )
            repulsion.setup(par.element, par.repulsion.effective)
            rep_energy = repulsion.get_engrad()

            # dispersion
            param = dict(a1=0.63, s8=2.4, a2=5.0)
            e_disp = d3.dftd3(mol.atomic_numbers, mol.positions, param)

            # partial charges
            eeq = charges.ChargeModel.param2019()
            _, qat = charges.solve(numbers, positions, charge, eeq, cn)

            return h, overlap_int, cn, rep_energy, e_disp, qat

        path = Path(self.path, out_name)

        # opening curly bracket for json
        with open(path, "w", encoding="utf-8") as f:
            f.write("{")

        # append each sample to json
        for i, (file, data) in enumerate(zip(self.file_list, self.data)):
            print(i, file)
            with open(path, "a", encoding="utf-8") as f:
                d = {}

                positions = torch.tensor(data[0], device=device, dtype=dtype)
                numbers = torch.tensor(data[1], device=device, dtype=torch.long)
                charge = torch.tensor(data[2], device=device, dtype=torch.int16)
                unpaired_e = torch.tensor(data[3], device=device, dtype=torch.int16)

                h, o, cnum, erep, edisp, qat = calc_singlepoint(
                    numbers, positions, charge, unpaired_e
                )

                d[file] = dict(
                    xyz=data[0],
                    numbers=data[1],
                    unpaired_e=data[3],
                    charges=data[2],
                    e_gfn1=[i * AU2KCAL for i in data[4]],
                    grad_gfn1=data[5],
                    e_ref=AU2KCAL * data[8],
                    grad_ref=data[9],
                    edisp=[i * AU2KCAL for i in edisp.tolist()],
                    erep=[i * AU2KCAL for i in erep.tolist()],
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
                if file != self.file_list[:-1]:
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

            energy, _ = read_tblite_gfn(path)

            if energy == 0 or not energy:
                raise ValueError(f"No energy found in {path}.")

            return energy

        reaction = {}
        count = 1

        path = Path(self.bpath, ".res")
        if not path.is_file():
            raise FileNotFoundError(f"File '{path}' not found.")

        with open(path, "r", encoding="utf-8") as f:
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
                        egfn1=e_gfn1 * AU2KCAL,
                        egfn2=e_gfn2 * AU2KCAL,
                        eref=ref,
                    )

                    count += 1

        with open(Path(self.bpath.parents[0], out_name), "w", encoding="utf-8") as f:
            json_dump_file(reaction, f)

    @staticmethod
    def get_dataloader(geometry, cfg: Optional[dict] = None) -> DataLoader:
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
