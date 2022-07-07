from json import load as json_load
from json import dump as json_dump
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union, overload

from xtbml.constants import FLOAT32
from xtbml.data.datareader import Datareader
from xtbml.typing import Tensor


class Sample:
    """Representation for single sample information."""

    buid: str
    """Unique identifier for the benchmark the sample is from."""
    uid: str
    """Unique identifier for sample"""
    numbers: Tensor
    """Atomic numbers"""
    positions: Tensor
    """Atomic positions"""
    unpaired_e: Tensor
    """Number of unpaired electrons"""
    charges: Tensor
    """Charge of sample"""
    gfn1_energy: Tensor
    """Atomwise energy calculated by GFN1-xTB"""
    gfn1_grad: Tensor
    """Gradient calculated by GFN1-xTB"""
    gfn2_energy: Tensor
    """Atomwise energy calculated by GFN2-xTB"""
    gfn2_grad: Tensor
    """Gradient calculated with GFN2-xTB"""
    dft_energy: Tensor
    """Reference DFT energy"""
    dft_grad: Tensor
    """Gradient calculated with DFT"""
    edisp: Tensor
    """Atomwise dispersion energy"""
    erep: Tensor
    """Atom-pairwise repulsion energy"""
    qat: Tensor
    """Atomwise partial charges (from EEQ)"""
    cn: Tensor
    """Atomwise coordination number"""
    ovlp: Tensor
    """Overlap matrix"""
    h0: Tensor
    """Hamiltonian matrix"""
    adj: Tensor
    """Adjacency matrix (optional)"""

    __slots__ = [
        "buid",
        "uid",
        "numbers",
        "positions",
        "unpaired_e",
        "charges",
        "gfn1_energy",
        "gfn1_grad",
        "gfn2_energy",
        "gfn2_grad",
        "dft_energy",
        "dft_grad",
        "edisp",
        "erep",
        "qat",
        "cn",
        "ovlp",
        "h0",
        "adj",
        "__device",
        "__dtype",
    ]

    # ...
    # TODO: add further QM features

    def __init__(
        self,
        buid: str,
        uid: str,
        numbers: Tensor,
        positions: Tensor,
        unpaired_e: Tensor,
        charges: Tensor,
        gfn1_energy: Tensor,
        gfn1_grad: Tensor,
        gfn2_energy: Tensor,
        gfn2_grad: Tensor,
        dft_energy: Tensor,
        dft_grad: Tensor,
        edisp: Tensor,
        erep: Tensor,
        qat: Tensor,
        cn: Tensor,
        h0: Tensor,
        ovlp: Tensor,
        adj: Tensor = Tensor([]),
    ) -> None:
        self.buid = buid
        self.uid = uid
        self.numbers = numbers
        self.positions = positions
        self.unpaired_e = unpaired_e
        self.charges = charges
        self.gfn1_energy = gfn1_energy
        self.gfn1_grad = gfn1_grad
        self.gfn2_energy = gfn2_energy
        self.gfn2_grad = gfn2_grad
        self.dft_energy = dft_energy
        self.dft_grad = dft_grad
        self.edisp = edisp
        self.erep = erep
        self.qat = qat
        self.cn = cn
        self.h0 = h0
        self.ovlp = ovlp
        self.adj = adj

        self.__device = self.positions.device
        self.__dtype = self.positions.dtype

        if any(
            [
                tensor.device != self.device
                for tensor in (
                    self.numbers,
                    self.positions,
                    self.unpaired_e,
                    self.charges,
                    self.gfn1_energy,
                    self.gfn1_grad,
                    self.gfn2_energy,
                    self.gfn2_grad,
                    self.dft_energy,
                    self.dft_grad,
                    self.edisp,
                    self.erep,
                    self.qat,
                    self.cn,
                    self.h0,
                    self.ovlp,
                    self.adj,
                )
            ]
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any(
            [
                tensor.dtype != self.dtype
                for tensor in (
                    self.positions,
                    self.gfn1_energy,
                    self.gfn1_grad,
                    self.gfn2_energy,
                    self.gfn2_grad,
                    self.dft_energy,
                    self.dft_grad,
                    self.edisp,
                    self.erep,
                    self.qat,
                    self.cn,
                    self.h0,
                    self.ovlp,
                    self.adj,
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
            self.buid,
            self.uid,
            self.positions.to(device=device),
            self.numbers.to(device=device),
            self.unpaired_e.to(device=device),
            self.charges.to(device=device),
            self.gfn1_energy.to(device=device),
            self.gfn1_grad.to(device=device),
            self.gfn2_energy.to(device=device),
            self.gfn2_grad.to(device=device),
            self.dft_energy.to(device=device),
            self.dft_grad.to(device=device),
            self.edisp.to(device=device),
            self.erep.to(device=device),
            self.qat.to(device=device),
            self.cn.to(device=device),
            self.h0.to(device=device),
            self.ovlp.to(device=device),
            self.adj.to(device=device),
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
            self.buid,
            self.uid,
            self.positions.type(dtype),
            self.numbers.type(torch.long),
            self.unpaired_e.type(torch.uint8),
            self.charges.type(torch.int8),
            self.gfn1_energy.type(dtype),
            self.gfn1_grad.type(dtype),
            self.gfn2_energy.type(dtype),
            self.gfn2_grad.type(dtype),
            self.dft_energy.type(dtype),
            self.dft_grad.type(dtype),
            self.edisp.type(dtype),
            self.erep.type(dtype),
            self.qat.type(dtype),
            self.cn.type(dtype),
            self.h0.type(dtype),
            self.ovlp.type(dtype),
            self.adj.type(dtype),
        )

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.uid})"

    def equal(self, other):
        if not isinstance(other, Sample):
            return NotImplemented

        return all(
            [
                self.uid == other.uid,
                torch.all(torch.isclose(self.positions, other.positions)).item(),
                torch.all(torch.isclose(self.numbers, other.numbers)).item(),
                torch.all(torch.isclose(self.unpaired_e, other.unpaired_e)).item(),
                torch.all(torch.isclose(self.charges, other.charges)).item(),
                torch.all(torch.isclose(self.gfn1_energy, other.gfn1_energy)).item(),
                torch.all(torch.isclose(self.gfn1_grad, other.gfn1_grad)).item(),
                torch.all(torch.isclose(self.gfn2_energy, other.gfn2_energy)).item(),
                torch.all(torch.isclose(self.gfn2_grad, other.gfn2_grad)).item(),
                torch.all(torch.isclose(self.dft_energy, other.dft_energy)).item(),
                torch.all(torch.isclose(self.dft_grad, other.dft_grad)).item(),
                torch.all(torch.isclose(self.edisp, other.edisp)).item(),
                torch.all(torch.isclose(self.erep, other.erep)).item(),
                torch.all(torch.isclose(self.qat, other.qat)).item(),
                torch.all(torch.isclose(self.cn, other.cn)).item(),
                torch.all(torch.isclose(self.h0, other.h0)).item(),
                torch.all(torch.isclose(self.ovlp, other.ovlp)).item(),
                torch.all(torch.isclose(self.adj, other.adj)).item(),
            ]
        )

    def to_dict(
        self, skipped: Optional[List[str]] = None
    ) -> Dict[str, Union[str, Tensor]]:
        """Create dictionary of class attributes (exluding dunder methods, `device`, `dtype` and callables).

        Args:
            skipped (List[str], optional): Attributes to exclude. Defaults to None.

        Returns:
            Dict[str, Union[str, Tensor]]: Selected attributes and their respective values.
        """
        d = {}
        skip = ["__", "device", "dtype"]
        if skipped is not None:
            skip = skip + skipped

        for slot in self.__slots__:
            if not any(s in slot for s in skip):
                d[slot] = getattr(self, slot)

        return d


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

        self.__device = samples[0].gfn1_energy.device
        self.__dtype = samples[0].gfn1_energy.dtype

    @classmethod
    def from_json(
        cls, path: Union[Path, str], dtype: torch.dtype = FLOAT32
    ) -> "Samples":
        """Create `Samples` from json.

        Parameters
        ----------
        path : Union[Path, str]
            Path to json file.
        dtype : torch.dtype, optional
            Type of the tensor. Defaults to FLOAT32.

        Returns
        -------
        Samples
            `Samples` instance that holds a list of `Sample`.
        """

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file '{path}' not found.")

        with open(path, "rb") as f:
            sample_list = []
            data = json_load(f)
            for uid, features in data.items():
                buid = uid.split(":")[0]

                # convert to tensor
                for feature, value in features.items():
                    features[feature] = torch.tensor(value, dtype=dtype)

                sample_list.append(Sample(buid=buid, uid=uid, **features))

        return cls(sample_list)

    @classmethod
    def from_disk(
        cls, benchmark: str, select_name: Union[str, None] = None
    ) -> "Samples":
        """Create `Samples` from disk. Features are set to zero.

        Parameters
        ----------
        benchmark : str
            Name of the benchmark.
        select_name : str, optional
            Name of the samples to select. Defaults to None.

        Returns
        -------
        Samples:
            `Samples` instance that holds a list of `Sample`.

        Examples
        --------
        >>> samples = Samples.from_disk("PTB")
        >>> print(samples)
        Samples(List of 12603 Sample objects)
        """

        data = Datareader(benchmark)
        data.get_sample_data()
        data.sort()
        if select_name is not None:
            data.get_by_name(select_name)

        samples_list = []
        for file, data in zip(data.file_list, data.data):
            samples_list.append(
                Sample(
                    buid=benchmark,
                    uid=file,
                    numbers=torch.tensor(data[0]),
                    positions=torch.tensor(data[1]),
                    charges=torch.tensor(data[2]),
                    unpaired_e=torch.tensor(data[3]),
                    gfn1_energy=torch.tensor(data[4]),
                    gfn1_grad=torch.tensor(data[5]),
                    gfn2_energy=torch.tensor(data[6]),
                    gfn2_grad=torch.tensor(data[7]),
                    dft_energy=torch.tensor(data[8]),
                    dft_grad=torch.tensor(data[9]),
                    edisp=torch.tensor(0.0),
                    erep=torch.tensor(0.0),
                    qat=torch.tensor(0.0),
                    cn=torch.tensor(0.0),
                    ovlp=torch.tensor(0.0),
                    h0=torch.tensor(0.0),
                ),
            )

        return cls(samples_list)

    def to_json(self, path: Union[Path, str]) -> None:
        """
        Convert samples to json.
        """
        d = {}
        for s in self.samples:
            d[s.uid] = s.to_dict()
            d[s.uid].pop("uid")

            # make tensor json serializable
            for k, v in d[s.uid].items():
                if isinstance(v, Tensor):
                    d[s.uid][k] = v.tolist()

        with open(Path(path, "samples.json"), "w", encoding="utf-8") as f:
            json_dump(d, f)

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

    @overload
    def __getitem__(self, idx: int) -> Sample:
        ...

    @overload
    def __getitem__(self, idx: slice) -> List[Sample]:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Sample, List[Sample]]:
        """Defines standard list slicing/indexing for list of samples."""
        return self.samples[idx]

    def __len__(self) -> int:
        """Defines length as number of samples in list of samples."""
        return len(self.samples)

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}(List of {len(self)} Sample objects)"
