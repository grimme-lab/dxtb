from json import load as json_load
from pathlib import Path

from typing import Dict, List, Optional, Union
from xtbml.typing import Tensor

import torch

from xtbml.adjlist import AdjacencyList
from xtbml.basis.type import get_cutoff
from xtbml.data.covrad import covalent_rad_d3
from xtbml.xtb.calculator import Calculator
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as gfn1_par
from xtbml.exlibs.tbmalt import Geometry
from xtbml.exlibs.tbmalt.batch import deflate
from xtbml.constants import FLOAT32


class Sample:
    """Representation for single sample information."""

    uid: str
    """Unique identifier for sample"""
    xyz: Tensor
    """Atomic positions"""
    numbers: Tensor
    """Atomic numbers"""
    unpaired_e: Tensor
    """Number of unpaired electrons"""
    charges: Tensor
    """Charge of sample"""
    egfn1: Tensor
    """Energy calculated by GFN1-xTB"""
    erep: Tensor
    """Atom-pairwise repulsion energy"""
    edisp: Tensor
    """Atom-wise dispersion energy"""
    ovlp: Tensor
    """Overlap matrix"""
    h0: Tensor
    """Hamiltonian matrix"""
    cn: Tensor
    """Coordination number"""

    __slots__ = [
        "uid",
        "xyz",
        "numbers",
        "unpaired_e",
        "charges",
        "egfn1",
        "edisp",
        "erep",
        "ovlp",
        "h0",
        "cn",
        "__device",
        "__dtype",
    ]

    # ...
    # TODO: add further QM features

    def __init__(
        self,
        uid: str,
        xyz: Tensor,
        numbers: Tensor,
        unpaired_e: Tensor,
        charges: Tensor,
        egfn1: Tensor,
        edisp: Tensor,
        erep: Tensor,
        ovlp: Tensor,
        h0: Tensor,
        cn: Tensor,
    ) -> None:
        self.uid = uid
        self.xyz = xyz
        self.numbers = numbers
        self.unpaired_e = unpaired_e
        self.charges = charges
        self.egfn1 = egfn1
        self.edisp = edisp
        self.erep = erep
        self.ovlp = ovlp
        self.h0 = h0
        self.cn = cn

        self.__device = xyz.device
        self.__dtype = xyz.dtype

        if any(
            [
                tensor.device != self.device
                for tensor in (
                    self.xyz,
                    self.numbers,
                    self.unpaired_e,
                    self.charges,
                    self.egfn1,
                    self.edisp,
                    self.erep,
                    self.ovlp,
                    self.h0,
                    self.cn,
                )
            ]
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any(
            [
                tensor.dtype != self.dtype
                for tensor in (
                    self.xyz,
                    self.egfn1,
                    self.edisp,
                    self.erep,
                    self.ovlp,
                    self.h0,
                    self.cn,
                )
            ]
        ):
            raise RuntimeError("All tensors must have the same dtype!")

    @property
    def device(self) -> torch.device:
        """The device on which the `Sample` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by Sample object."""
        return self.__dtype

    def to(self, device: torch.device) -> "Sample":
        """
        Returns a copy of the `Sample` instance on the specified device.
        This method creates and returns a new copy of the `Sample` instance
        on the specified device "``device``".
        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.
        Returns
        -------
        Sample
            A copy of the `Sample` instance placed on the specified device.
        Notes
        -----
        If the `Sample` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.uid,
            self.xyz.to(device=device),
            self.numbers.to(device=device),
            self.unpaired_e.to(device=device),
            self.charges.to(device=device),
            self.egfn1.to(device=device),
            self.edisp.to(device=device),
            self.erep.to(device=device),
            self.ovlp.to(device=device),
            self.h0.to(device=device),
            self.cn.to(device=device),
        )

    def type(self, dtype: torch.dtype) -> "Sample":
        """
        Returns a copy of the `Sample` instance with specified floating point type.
        This method creates and returns a new copy of the `Sample` instance
        with the specified dtype.
        Parameters
        ----------
        dtype : torch.dtype
            Type of the
        Returns
        -------
        Sample
            A copy of the `Sample` instance with the specified dtype.
        Notes
        -----
        If the `Sample` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.uid,
            self.xyz.type(dtype),
            self.numbers.type(torch.long),
            self.unpaired_e.type(torch.uint8),
            self.charges.type(torch.uint8),
            self.egfn1.type(dtype),
            self.edisp.type(dtype),
            self.erep.type(dtype),
            self.ovlp.type(dtype),
            self.h0.type(dtype),
            self.cn.type(dtype),
        )

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.uid})"

    def to_dict(
        self, skipped: Optional[List[str]] = None
    ) -> Dict[str, Union[str, Tensor]]:
        """Create dictionary of class attributes (exluding dunder methods, `device`, `dtype` and callables).

        Args:
            skipped (List[str], optional): Attributes to exclude. Defaults to None.

        Returns:
            Dict[str, Union[str, Tensor]]: Selected attributes and their respective values.
        """
        d = dict()
        skip = ["__", "device", "dtype"]
        if skipped is not None:
            skip = skip + skipped

        for slot in self.__slots__:
            if not any([s in slot for s in skip]):
                d[slot] = getattr(self, slot)

        return d

    def calc_singlepoint(self) -> List[Tensor]:
        """Calculate QM features based on classicals input, such as geometry."""

        # CALC FEATURES

        # NOTE: singlepoint calculation so far only working with no padding
        # quick and dirty hack to get rid of padding
        mol = Geometry(
            atomic_numbers=deflate(self.numbers),
            positions=deflate(self.xyz, axis=1),
            charges=self.charges,
            unpaired_e=self.unpaired_e,
        )

        # check underlying data consistent
        assert (
            self.numbers.storage().data_ptr() == mol.atomic_numbers.storage().data_ptr()
        )

        # setup calculator
        par = gfn1_par
        calc = Calculator(mol, par)

        # prepate cutoffs and lattice
        # trans = get_lattice_points(mol_periodic, mol_lattice, cn_cutoff)
        rcov = covalent_rad_d3
        cn = get_coordination_number(mol, rcov, exp_count)

        cutoff = get_cutoff(calc.basis)
        # trans = get_lattice_points(mol_periodic, mol_lattice, cutoff)
        trans = torch.zeros([3, 1])
        adj = AdjacencyList(mol, trans, cutoff)

        # build hamiltonian
        h, overlap_int = calc.hamiltonian.build(calc.basis, adj, cn)

        return h, overlap_int, cn


class Samples:
    """Representation for list of samples."""

    samples: List[Sample]
    """List of samples"""

    __slots__ = [
        "samples",
        "__device",
        "__dtype",
    ]

    def __init__(self, samples: List[Sample]):
        self.samples = samples

        self.__device = samples[0].egfn1.device
        self.__dtype = samples[0].egfn1.dtype

    @classmethod
    def from_json(
        cls, path: Union[Path, str], dtype: torch.dtype = FLOAT32
    ) -> "Samples":
        """Create `Samples` from json.

        Args:
            path (Union[Path, str]): Path of JSON file.

        Raises:
            FileNotFoundError: Error if JSON file not found.

        Returns:
            Samples: Class that holds a list of `Sample`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file '{path}' not found.")

        with open(path, "rb") as f:
            sample_list = []
            data = json_load(f)
            for uid, features in data.items():
                # convert to tensor
                for feature, value in features.items():
                    features[feature] = torch.tensor(value, dtype=dtype)

                # TODO: add values to constructor
                if "ees" in features:
                    features.pop("ees", None)
                if "qat" in features:
                    features.pop("qat", None)

                sample_list.append(Sample(uid=uid, **features))

        return cls(sample_list)

    def to_json(self) -> Dict:
        """
        Convert sample to json.
        """
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """The device on which the `Samples` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by Samples object."""
        return self.__dtype

    def to(self, device: torch.device) -> "Samples":
        """
        Returns a copy of the `Samples` instance on the specified device.
        This method creates and returns a new copy of the `Samples` instance
        on the specified device "``device``".
        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.
        Returns
        -------
        Samples
            A copy of the `Samples` instance placed on the specified device.
        Notes
        -----
        If the `Samples` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__([sample.to(device=device) for sample in self.samples])

    def type(self, dtype: torch.dtype) -> "Samples":
        """
        Returns a copy of the `Samples` instance with specified floating point type.
        This method creates and returns a new copy of the `Samples` instance
        with the specified dtype.
        Parameters
        ----------
        dtype : torch.dtype
            Type of the
        Returns
        -------
        Samples
            A copy of the `Samples` instance with the specified dtype.
        Notes
        -----
        If the `Samples` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__([sample.type(dtype) for sample in self.samples])

    def __getitem__(self, idx) -> Union[Sample, List[Sample]]:
        """Defines standard list slicing/indexing for list of samples."""
        return self.samples[idx]

    def __len__(self) -> int:
        """Defines length as number of samples in list of samples."""
        return len(self.samples)

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}(List of {len(self)} Sample objects)"
