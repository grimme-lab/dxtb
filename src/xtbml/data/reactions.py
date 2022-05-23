from json import load as json_load
from json import dump as json_dump
from pathlib import Path

from typing import Dict, List, Optional, Union, overload
from xtbml.typing import Tensor

import torch


class Reaction:
    """Representation for single reaction involving multiple `Reaction`s."""

    uid: str
    """Unique identifier for reaction"""
    partners: List[str]
    """List of participants in the reaction"""
    nu: Tensor
    """Stoichiometry coefficient for respective participant"""
    egfn1: Tensor
    """Reaction energies given by GFN1-xTB"""
    egfn2: Tensor
    """Reaction energies given by GFN2-xTB"""
    eref: Tensor
    """Reaction energies given by reference method"""

    __slots__ = [
        "uid",
        "partners",
        "nu",
        "egfn1",
        "egfn2",
        "eref",
        "__device",
        "__dtype",
    ]

    def __init__(
        self,
        uid: str,
        partners: List[str],
        nu: Tensor,
        egfn1: Tensor,
        egfn2: Tensor,
        eref: Tensor,
    ) -> None:
        self.uid = uid
        self.partners = partners
        self.nu = nu
        self.egfn1 = egfn1
        self.egfn2 = egfn2
        self.eref = eref

        self.__dtype = egfn1.dtype
        self.__device = egfn1.device

        if any(
            [
                tensor.device != self.device
                for tensor in (self.nu, self.egfn1, self.egfn2, self.eref)
            ]
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any(
            [
                tensor.dtype != self.dtype
                for tensor in (self.egfn1, self.egfn2, self.eref)
            ]
        ):
            raise RuntimeError("All tensors must have the same dtype!")

    @property
    def device(self) -> torch.device:
        """The device on which the `Reaction` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by Reaction object."""
        return self.__dtype

    def to(self, device: torch.device) -> "Reaction":
        """
        Returns a copy of the `Reaction` instance on the specified device.
        This method creates and returns a new copy of the `Reaction` instance
        on the specified device "``device``".
        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.
        Returns
        -------
        Reaction
            A copy of the `Reaction` instance placed on the specified device.
        Notes
        -----
        If the `Reaction` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__(
            self.uid,
            self.partners,
            self.nu.to(device=device),
            self.egfn1.to(device=device),
            self.egfn2.to(device=device),
            self.eref.to(device=device),
        )

    def type(self, dtype: torch.dtype) -> "Reaction":
        """
        Returns a copy of the `Reaction` instance with specified floating point type.
        This method creates and returns a new copy of the `Reaction` instance
        with the specified dtype.
        Parameters
        ----------
        dtype : torch.dtype
            Type of the
        Returns
        -------
        Reaction
            A copy of the `Reaction` instance with the specified dtype.
        Notes
        -----
        If the `Reaction` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__(
            self.uid,
            self.partners,
            self.nu.type(torch.int8),
            self.egfn1.type(dtype),
            self.egfn2.type(dtype),
            self.eref.type(dtype),
        )

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.uid})"

    def equal(self, other):
        if not isinstance(other, Reaction):
            return NotImplemented

        return all(
            [
                self.uid == other.uid,
                self.partners == other.partners,
                torch.equal(self.nu, other.nu),
                torch.equal(self.egfn1, other.egfn1),
                torch.equal(self.egfn2, other.egfn2),
                torch.equal(self.eref, other.eref),
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
        d = dict()
        skip = ["__", "device", "dtype"]
        if skipped is not None:
            skip = skip + skipped

        for slot in self.__slots__:
            if not any([s in slot for s in skip]):
                d[slot] = getattr(self, slot)

        return d


class Reactions:
    """Representation for list of `Reaction`s."""

    reactions: List[Reaction]
    """List of reactions"""

    def __init__(self, reactions) -> None:
        self.reactions = reactions

        self.__device = reactions[0].egfn1.device
        self.__dtype = reactions[0].egfn1.dtype

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> "Reactions":
        """Create `Reactions` from json.

        Args:
            path (Union[Path, str]): Path of JSON file to read.

        Raises:
            FileNotFoundError: Error if JSON file not found.

        Returns:
            Reactions: Class that holds a list of `Reaction`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"JSON file '{path}' not found.")

        with open(path, "rb") as f:
            reaction_list = []
            data = json_load(f)
            for uid, features in data.items():
                # convert to tensor
                features["nu"] = torch.tensor(features["nu"])
                features["egfn1"] = torch.tensor(features["egfn1"])
                features["egfn2"] = torch.tensor(features["egfn2"])
                features["eref"] = torch.tensor(features["eref"])

                reaction_list.append(Reaction(uid=uid, **features))

        return cls(reaction_list)

    def to_json(self, path: Union[Path, str]) -> None:
        """
        Convert reactions to json.
        """
        d = {}
        for r in self.reactions:
            d[r.uid] = r.to_dict()
            d[r.uid].pop("uid")

            # make tensor json serializable
            for k, v in d[r.uid].items():
                if isinstance(v, torch.Tensor):
                    try:
                        d[r.uid][k] = v.item()
                    except ValueError:
                        d[r.uid][k] = v.tolist()

        with open(Path(path, "reactions.json"), "w") as f:
            json_dump(d, f)

    def __len__(self) -> int:
        """Defines length as number of reactions in list of reactions."""
        return len(self.reactions)

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}(List of {len(self)} Reaction objects)"

    @overload
    def __getitem__(self, idx: int) -> Reaction:
        ...

    @overload
    def __getitem__(self, idx: slice) -> List[Reaction]:
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union[Reaction, List[Reaction]]:
        """Defines standard list slicing/indexing for list of reactions."""
        return self.reactions[idx]

    @property
    def device(self) -> torch.device:
        """The device on which the `Reactions` object resides."""
        return self.__device

    @device.setter
    def device(self, *args):
        """Instruct users to use the ".to" method if wanting to change device."""
        raise AttributeError("Move object to device using the `.to` method")

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype used by Reactions object."""
        return self.__dtype

    def to(self, device: torch.device) -> "Reactions":
        """
        Returns a copy of the `Reactions` instance on the specified device.
        This method creates and returns a new copy of the `Reactions` instance
        on the specified device "``device``".
        Parameters
        ----------
        device : torch.device
            Device to which all associated tensors should be moved.
        Returns
        -------
        Reactions
            A copy of the `Reactions` instance placed on the specified device.
        Notes
        -----
        If the `Reactions` instance is already on the desired device `self` will be returned.
        """
        if self.__device == device:
            return self

        return self.__class__([sample.to(device=device) for sample in self.reactions])

    def type(self, dtype: torch.dtype) -> "Reactions":
        """
        Returns a copy of the `Reactions` instance with specified floating point type.
        This method creates and returns a new copy of the `Reactions` instance
        with the specified dtype.
        Parameters
        ----------
        dtype : torch.dtype
            Type of the
        Returns
        -------
        Reactions
            A copy of the `Reactions` instance with the specified dtype.
        Notes
        -----
        If the `Reactions` instance has already the desired dtype `self` will be returned.
        """
        if self.__dtype == dtype:
            return self

        return self.__class__([reaction.type(dtype) for reaction in self.reactions])
