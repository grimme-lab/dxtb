from json import load as json_load
from json import dump as json_dump
from pathlib import Path
import torch
from typing import Dict, List, Optional, Union, overload

from xtbml.constants import FLOAT32
from xtbml.typing import Tensor


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
    """Atomwise energy calculated by GFN1-xTB"""
    egfn2: Tensor
    """Atomwise energy calculated by GFN2-xTB"""
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

    __slots__ = [
        "uid",
        "xyz",
        "numbers",
        "unpaired_e",
        "charges",
        "egfn1",
        "egfn2",
        "edisp",
        "erep",
        "qat",
        "cn",
        "h0",
        "ovlp",
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
        egfn2: Tensor,
        edisp: Tensor,
        erep: Tensor,
        qat: Tensor,
        cn: Tensor,
        h0: Tensor,
        ovlp: Tensor,
    ) -> None:
        self.uid = uid
        self.xyz = xyz
        self.numbers = numbers
        self.unpaired_e = unpaired_e
        self.charges = charges
        self.egfn1 = egfn1
        self.egfn2 = egfn2
        self.edisp = edisp
        self.erep = erep
        self.qat = qat
        self.cn = cn
        self.h0 = h0
        self.ovlp = ovlp

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
                    self.egfn2,
                    self.edisp,
                    self.erep,
                    self.qat,
                    self.cn,
                    self.h0,
                    self.ovlp,
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
                    self.egfn2,
                    self.edisp,
                    self.erep,
                    self.qat,
                    self.cn,
                    self.h0,
                    self.ovlp,
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
            self.egfn2.to(device=device),
            self.edisp.to(device=device),
            self.erep.to(device=device),
            self.qat.to(device=device),
            self.cn.to(device=device),
            self.h0.to(device=device),
            self.ovlp.to(device=device),
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
            self.egfn2.type(dtype),
            self.edisp.type(dtype),
            self.erep.type(dtype),
            self.qat.type(dtype),
            self.cn.type(dtype),
            self.h0.type(dtype),
            self.ovlp.type(dtype),
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
                torch.all(torch.isclose(self.xyz, other.xyz)).item(),
                torch.all(torch.isclose(self.numbers, other.numbers)).item(),
                torch.all(torch.isclose(self.unpaired_e, other.unpaired_e)).item(),
                torch.all(torch.isclose(self.charges, other.charges)).item(),
                torch.all(torch.isclose(self.egfn1, other.egfn1)).item(),
                torch.all(torch.isclose(self.egfn2, other.egfn2)).item(),
                torch.all(torch.isclose(self.edisp, other.edisp)).item(),
                torch.all(torch.isclose(self.erep, other.erep)).item(),
                torch.all(torch.isclose(self.qat, other.qat)).item(),
                torch.all(torch.isclose(self.cn, other.cn)).item(),
                torch.all(torch.isclose(self.h0, other.h0)).item(),
                torch.all(torch.isclose(self.ovlp, other.ovlp)).item(),
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

                sample_list.append(Sample(uid=uid, **features))

        return cls(sample_list)

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

        with open(Path(path, "samples.json"), "w") as f:
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
