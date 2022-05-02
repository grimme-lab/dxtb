from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel
from pathlib import Path
from json import load as json_load

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from xtbml.adjlist import AdjacencyList
from xtbml.basis.type import get_cutoff
from xtbml.data.covrad import covalent_rad_d3
from xtbml.xtb.calculator import Calculator
from xtbml.ncoord.ncoord import get_coordination_number, exp_count
from xtbml.param.gfn1 import GFN1_XTB as gfn1_par
from xtbml.exlibs.tbmalt import Geometry
from xtbml.exlibs.tbmalt.batch import deflate


##########################################################################################
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
    """Energy calculated by GFN1-xtb"""
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
            self.ovlp.type(dtype),
            self.h0.type(dtype),
            self.cn.type(dtype),
        )

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.uid})"

    def dict(
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
    def from_json(cls, path: Union[Path, str]) -> "Samples":
        """Create samples from json.

        Args:
            path (Union[Path, str]): Path of JSON file.

        Raises:
            FileNotFoundError: Error if JSON file not found.

        Returns:
            Samples: Class that holds a list of `Sample`.
        """
        if type(path) == str:
            path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"JSON file '{path}' not found.")

        with open(path, "rb") as f:
            sample_list = []
            data = json_load(f)
            for uid, features in data.items():
                # convert to tensor
                for feature, value in features.items():
                    # print(features[feature], value)
                    features[feature] = torch.tensor(value)

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


class Reaction:
    """Representation for single reaction involving multiple `Reaction`s."""

    uid: str
    """Unique identifier for reaction"""
    partners: List[str]
    """List of participants in the reaction"""
    nu: Tensor
    """Stoichiometry coefficient for respective participant"""
    egfn1: Tensor
    """Reaction energies given by GFN1-xtb"""
    eref: Tensor
    """Reaction energies given by reference method"""

    __slots__ = [
        "uid",
        "partners",
        "nu",
        "egfn1",
        "eref",
        "__device",
        "__dtype",
    ]

    def __init__(
        self, uid: str, partners: List[str], nu: Tensor, egfn1: Tensor, eref: Tensor
    ) -> None:
        self.uid = uid
        self.partners = partners
        self.nu = nu
        self.egfn1 = egfn1
        self.eref = eref

        self.__dtype = egfn1.dtype
        self.__device = egfn1.device

        if any(
            [
                tensor.device != self.device
                for tensor in (self.nu, self.egfn1, self.eref)
            ]
        ):
            raise RuntimeError("All tensors must be on the same device!")

        if any([tensor.dtype != self.dtype for tensor in (self.egfn1, self.eref)]):
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
            self.eref.type(dtype),
        )

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}({self.uid})"

    def dict(
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
    """Representation for list of samples."""

    reactions: List[Reaction]
    """List of reactions"""

    def __init__(self, reactions) -> None:
        self.reactions = reactions

        self.__device = reactions[0].egfn1.device
        self.__dtype = reactions[0].egfn1.dtype

    @classmethod
    def from_json(cls, path: Union[Path, str]) -> "Reactions":
        """Create samples from json.

        Args:
            path (Union[Path, str]): Path of JSON file to read.

        Raises:
            FileNotFoundError: Error if JSON file not found.

        Returns:
            Reactions: Class that holds a list of `Reaction`.
        """
        if type(path) == str:
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
                features["eref"] = torch.tensor(features["eref"])

                reaction_list.append(Reaction(uid=uid, **features))

        return cls(reaction_list)

    def to_json(self) -> Dict:
        """
        Convert reaction to json.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Defines length as number of reactions in list of reactions."""
        return len(self.reactions)

    def __repr__(self) -> str:
        """Custom print representation of class."""
        return f"{self.__class__.__name__}(List of {len(self)} Reaction objects)"

    def __getitem__(self, idx) -> Union[Reaction, List[Reaction]]:
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


##########################################################################################


class ReactionDataset(BaseModel, Dataset):
    """Dataset for storing features used for training."""

    # TODO: better would be an object of lists than a list of objects
    samples: List[Sample]
    """Samples in dataset"""
    reactions: List[Reaction]
    """Reactions in dataset"""

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def create_from_disk(
        path_reactions: Union[Path, str], path_samples: Union[Path, str]
    ) -> "ReactionDataset":
        """Load `Samples` and `Reactions` from JSON files.

        Args:
            path_reactions (str): Path of JSON file for reactions.
            path_samples (str): Path of JSON file for samples.

        Returns:
            ReactionDataset: Dataset for storing features used for training
        """
        reactions = Reactions.from_json(path_reactions)
        samples = Samples.from_json(path_samples)

        return ReactionDataset(samples=samples.samples, reactions=reactions.reactions)

    def __len__(self):
        """Length of dataset defined by number of reactions."""
        return len(self.reactions)

    def __getitem__(self, idx: int):
        """Get all samples involved in specified reaction."""
        reaction = self.reactions[idx]
        samples = [s for s in self.samples if s.uid in reaction.partners]
        if samples == []:
            print(f"WARNING: Samples for reaction {reaction} not available")
        return samples, reaction

    def rm_reaction(self, idx: int):
        """Remove reaction from dataset."""
        # NOTE: Dataset might contain samples
        #       that are not required in any reaction anymore
        del self.reactions[idx]

    def get_dataloader(self, cfg: Optional[dict] = {}) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over dataset object.

        Args:
            cfg (dict, optional): Optional configuration for dataloader settings.

        Returns:
            DataLoader: Pytorch dataloader
        """

        def collate_fn(batch) -> Tuple[List[Sample], List[Reaction]]:
            # NOTE: for first setup simply take in list of (samples, reaction) tuples
            # TODO: this does not parallelize well, for correct handling, tensorwise
            #       concatenation of samples and reactions properties is necessary

            # fixed number of partners
            # assert all([len(s[0]) == len(batch[0][0]) for s in batch])

            batched_samples = [{} for _ in range(len(batch[0][0]))]
            batched_reactions = []

            for i, s in enumerate(batch):
                # assume standardised features
                """assert all(
                    sample.positions.shape == s[0][0].positions.shape for sample in s[0]
                )
                assert all(
                    sample.atomic_numbers.shape == s[0][0].atomic_numbers.shape
                    for sample in s[0]
                )
                assert all(
                    sample.overlap.shape == s[0][0].overlap.shape for sample in s[0]
                )
                assert all(
                    sample.hamiltonian.shape == s[0][0].hamiltonian.shape
                    for sample in s[0]
                )"""  # TODO: as far as padding missing

                # batch samples
                for j, sample in enumerate(s[0]):
                    # print(sample.uid)
                    if i == 0:
                        batched_samples[j] = sample.to_dict()
                        for k, v in batched_samples[j].items():
                            if isinstance(v, Tensor):
                                batched_samples[j][k] = v.unsqueeze(0)
                                # print(k, v.shape)
                        continue

                    for k, v in sample.to_dict().items():
                        if not isinstance(v, Tensor):
                            continue
                        batched_samples[j][k] = torch.concat(
                            (batched_samples[j][k], v.unsqueeze(0)), dim=0
                        )

                # batch reactions
                batched_reactions.append(s[1])
                # NOTE: item i belongs to features in batched_samples[j][i],
                # with j being the index of reactant

            # TODO: could be added as an _add_ function to the class
            partners = [i for r in batched_reactions for i in r.partners]

            # TODO: requires reactions with same number of partners
            assert (
                len(set([len(r.partners) for r in batched_reactions])) == 1
            ), "Number of partners must be the same for all reactions (until padding or batching is implemented)"

            nu = torch.stack([r.nu for r in batched_reactions], 0)
            egfn1 = torch.stack([r.egfn1 for r in batched_reactions], 0)
            eref = torch.stack([r.eref for r in batched_reactions], 0)

            # NOTE: Example on how batching of Reaction objects is conducteds
            # [Reaction(uid='AB', partners=['A', 'B', 'AB'], nu=[-1, -1, 1],
            # egfn1=tensor([1.2300]), eref=tensor([1.5400])),
            # Reaction(uid='AC', partners=['A', 'C', 'AC'], nu=[-1, -1, 1],
            # egfn1=tensor([3.4500]), eref=tensor([7.2300]))]
            # -->
            # Reaction(uid='BATCH', partners=[['A', 'B', 'AB'], ['A', 'C', 'AC']], nu=[[-1, -1, 1], [-1, -1, 1]],
            # egfn1=tensor([[1.2300], [3.4500]]), eref=tensor([[1.5400], [7.2300]])),

            # convert to sample objects (optional)
            batched_samples = [Sample(**d) for d in batched_samples]
            batched_reactions = Reaction(
                uid="BATCH", partners=partners, nu=nu, egfn1=egfn1, eref=eref
            )

            # NOTE: information on how samples and shapes are aligned
            # [sampleA, sampleB, sampleC, ...] <-- each sample has same features which have identical shapes respectively
            # a, b, c, (d)
            # a.feature1.shape == (bs, feature1_size, ...)
            # a.feature2.shape == (bs, feature2_size, ...)
            # ...
            # b.feature1.shape == (bs, feature1_size, ...)

            # A, B, AB, A, C, AC
            # --> batched_samples[0] == A,A, 1 == B,C, 2 == AB,AC
            # print(batched_samples[1].hamiltonian)  # B + C

            return batched_samples, batched_reactions

        if "collate_fn" not in cfg:
            cfg["collate_fn"] = collate_fn

        return DataLoader(self, **cfg)

    def pad(self):
        """Conduct padding on all samples in the dataset."""

        def get_max_shape(
            dataset: ReactionDataset,
        ) -> List[Union[None, Tuple[int, int]]]:
            # ordered list containing each key
            features = list(dataset.samples[0].to_dict().keys())
            max_shape = [None for f in features]

            # get max length/shape for each feature respectively
            for s in dataset.samples:
                for i, k in enumerate(features):
                    f = getattr(s, k)
                    if not isinstance(f, Tensor):
                        continue
                    sh = f.shape
                    if max_shape[i] is None:
                        max_shape[i] = list(sh)
                    else:
                        for j in range(len(sh)):
                            max_shape[i][j] = max(max_shape[i][j], sh[j])

            return max_shape

        # get maximal shape for each feature
        max_shape = get_max_shape(self)

        # ordered list containing each key
        features = list(self.samples[0].to_dict().keys())

        # pad all features to max lengths
        for s in self.samples:
            for i, k in enumerate(features):
                f = getattr(s, k)
                if not isinstance(f, Tensor):
                    continue
                sh = f.shape
                for j in range(len(sh)):
                    abc = max_shape[i][j] > sh[j]
                    if abc:
                        # pad jth-dimension to max shape
                        pad = [0 for _ in range(2 * len(sh))]
                        # change respective entry to difference
                        idx = 2 * (len(sh) - j) - 1
                        pad[idx] = max_shape[i][j] - sh[j]
                        f = torch.nn.functional.pad(f, pad, mode="constant", value=0.0)
                        setattr(s, k, f)
        return
