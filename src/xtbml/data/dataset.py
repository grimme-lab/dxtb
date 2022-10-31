from abc import abstractmethod
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union, overload

import pandas as pd
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from xtbml.data.reactions import Reaction, Reactions
from xtbml.data.samples import Sample, Samples


# TODO: add to general utils
# With courtesy to https://pytorch-forecasting.readthedocs.io
def padded_stack(
    tensors: list[torch.Tensor],
    side: str = "right",
    mode: str = "constant",
    value: Union[int, float] = 0,
) -> torch.Tensor:
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.

    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding

    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value)
            if full_size - x.size(-1) > 0
            else x
            for x in tensors
        ],
        dim=0,
    )
    return out


def get_subsets_from_batched_reaction(batched_reaction: Reaction) -> list[str]:
    # derive subset from partner list
    subsets = [s.split("/")[0] for s in batched_reaction.partners]
    # different number of partners per reaction
    n_partner = torch.count_nonzero(batched_reaction.nu, dim=1)
    # get reaction-partner idx
    p_idx = torch.cumsum(n_partner, dim=0) - 1
    # contract labels
    label = [subsets[i] for i in p_idx]
    return label


class DatasetModel(BaseModel, Dataset):
    """Base class for Datasets."""

    # TODO: better would be an object of lists than a list of objects
    samples: list[Sample]
    """Samples in dataset"""

    reactions: Optional[list[Reaction]] = None
    """Reactions in dataset"""

    class Config:
        arbitrary_types_allowed = True

    @overload
    @classmethod  # @classmethod must be used before @abstractmethod!
    @abstractmethod
    def from_json(
        cls, path_samples: Union[Path, list[Path], str, list[str]]
    ) -> "SampleDataset":
        ...

    @overload
    @classmethod
    @abstractmethod
    def from_json(
        cls, path_samples: Union[Path, str], path_reactions: Union[Path, str]
    ) -> "ReactionDataset":
        ...

    @classmethod
    @abstractmethod
    def from_json(
        cls,
        path_samples: Union[Path, list[Path], str, list[str]],
        path_reactions: Union[Path, list[Path], str, list[str], None] = None,
    ) -> Union["SampleDataset", "ReactionDataset"]:
        """Load `Samples` from JSON files.

        Parameters
        ----------
        path_samples : Union[Path, List[Path], str, List[str]]
            Path of JSON file for samples.
        path_reactions : Union[Path, List[Path], str, List[str], None]
            Path of JSON file for reactions.

        Returns
        -------
        Union[SampleDataset, ReactionDataset]
            Dataset for storing features used for training.
        """
        ...

    def to_json(self, path: Union[Path, str]) -> None:
        """Save dataset to disk.

        Parameters
        ----------
        path : Union[Path, str]
            Path to save dataset to.
        """
        Samples(samples=self.samples).to_json(path)
        if self.reactions is not None:
            Reactions(reactions=self.reactions).to_json(path)

    @classmethod
    def merge(cls, a, b):
        """Merge two datasets."""
        return cls(samples=a.samples + b.samples, reactions=a.reactions + b.reactions)

    def __len__(self):
        """Length of dataset defined by number of reactions or samples."""
        return len(self.samples) if self.reactions is None else len(self.reactions)


class SampleDataset(DatasetModel):
    """Dataset for storing features used for training."""

    samples: list[Sample]
    """Samples in dataset"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_json(
        cls, path_samples: Union[Path, list[Path], str, list[str]]
    ) -> "SampleDataset":
        if isinstance(path_samples, list):
            sample_list = []
            for path in path_samples:
                samples = Samples.from_json(path).samples
                sample_list.extend(samples)
        else:
            sample_list = Samples.from_json(path_samples).samples

        return cls(samples=sample_list)

    def __eq__(self, other: "SampleDataset") -> bool:
        """Compare `SampleDataset` to another one.

        Parameters
        ----------
        other : SampleDataset
            Dataset to compare to.

        Returns
        -------
        bool
            Result of comparison.

        Raises
        ------
        NotImplementedError
            If other is not of type `SampleDataset`.
        """
        if not isinstance(other, SampleDataset):
            raise NotImplementedError(
                "Comparison with other types than `SampleDataset` not possible."
            )

        for i, sample in enumerate(self.samples):
            if not sample.equal(other.samples[i]):
                return False

        return True

    @overload
    def __getitem__(self, idx: int) -> "Sample":
        ...

    @overload
    def __getitem__(self, idx: slice) -> "SampleDataset":
        ...

    def __getitem__(self, idx: Union[int, slice]) -> Union["Sample", "SampleDataset"]:
        """Defines standard list slicing/indexing for list of `SampleDataset`."""
        samples = self.samples[idx]

        if isinstance(idx, slice) and isinstance(samples, list):
            return SampleDataset(samples=self.samples[idx])

        if isinstance(idx, int) and isinstance(samples, Sample):
            return samples

        raise TypeError(f"Invalid index '{idx}' type.")


class ReactionDataset(DatasetModel):
    """Dataset for storing features used for training."""

    samples: list[Sample]
    """Samples in dataset"""

    reactions: list[Reaction]
    """Reactions in dataset"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_json(
        cls,
        path_samples: Union[Path, list[Path], str, list[str]],
        path_reactions: Union[Path, list[Path], str, list[str]],
    ) -> "ReactionDataset":
        if isinstance(path_samples, list):
            sample_list = []
            for path in path_samples:
                samples = Samples.from_json(path).samples
                sample_list.extend(samples)
        else:
            sample_list = Samples.from_json(path_samples).samples

        if isinstance(path_reactions, list):
            reaction_list = []
            for path in path_reactions:
                reactions = Reactions.from_json(path).reactions
                reaction_list.extend(reactions)
        else:
            reaction_list = Reactions.from_json(path_reactions).reactions

        return cls(samples=sample_list, reactions=reaction_list)

    def get_samples_from_reaction_partners(self, reaction: Reaction) -> list[Sample]:
        sample_list = []
        for partner in reaction.partners:
            for sample in self.samples:
                if sample.uid == partner:
                    sample_list.append(sample)

        return sample_list

    @overload
    def __getitem__(self, idx: int) -> tuple[list[Sample], Reaction]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "ReactionDataset":
        ...

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[tuple[list[Sample], Reaction], "ReactionDataset"]:
        """Get all samples involved in specified reaction."""

        reactions = self.reactions[idx]

        if isinstance(idx, slice) and isinstance(reactions, list):
            samples = [
                self.get_samples_from_reaction_partners(reaction)
                for reaction in reactions
            ]

            return ReactionDataset(
                samples=list({item for sublist in samples for item in sublist}),
                reactions=reactions,
            )
        elif isinstance(idx, int) and isinstance(reactions, Reaction):
            # NOTE:
            # The order of partners in `List[Sample]` is preserved.
            # It gets messed up when slicing and removing duplicates.
            #
            # The following does NOT preserve the order:
            # `[s for s in self.samples if s.uid in reactions.partners]`

            samples = self.get_samples_from_reaction_partners(reactions)

            if len(samples) == 0:
                # TODO: Use errir in production
                # raise RuntimeError(f"WARNING: No samples found for reaction {reactions}.")
                print(f"WARNING: No samples found for reaction {reactions}.")

            return samples, reactions
        else:
            raise TypeError(f"Invalid index '{idx}' type.")

    # FIXME: Not needed currently -> remove?
    def rm_reaction(self, idx: int):
        """Remove reaction from dataset."""
        # NOTE: Dataset might contain samples
        #       that are not required in any reaction anymore
        del self.reactions[idx]

    def copy(self) -> "ReactionDataset":
        """Return a copy of the dataset."""
        return ReactionDataset(samples=self.samples, reactions=self.reactions)

    def __eq__(self, other: "ReactionDataset") -> bool:
        """Compare `ReactionsDataset` to another one.

        Parameters
        ----------
        other : ReactionDataset
            Dataset to compare to.

        Returns
        -------
        bool
            Result of comparison.

        Raises
        ------
        NotImplementedError
            If other is not of type `ReactionDataset`.

        Note
        ----
        For the `ReactionDataset`s to be equal, the order of `ReactionDatasets.samples` and `ReactionDatasets.reactions` must be equal (see `ReactionDatasets.sort("both")`).
        """
        if not isinstance(other, ReactionDataset):
            raise NotImplementedError(
                "Comparison with other types than `ReactionDataset` not possible."
            )

        for i, sample in enumerate(self.samples):
            if not sample.equal(other.samples[i]):
                return False

        for i, reaction in enumerate(self.reactions):
            if not reaction.equal(other.reactions[i]):
                return False

        return True

    def sort(self, target: Literal["samples", "reactions", "both"] = "samples") -> None:
        """Sort the samples and reactions in the dataset by their unique identifiers (UIDs).

        Parameters
        ----------
        target : Literal[samples, reactions, both], optional
            Selects what will be sorted, by default "samples"
        """

        @overload
        def _sort(sort_target: list[Sample]) -> list[Sample]:
            ...

        @overload
        def _sort(sort_target: list[Reaction]) -> list[Reaction]:
            ...

        def _sort(
            sort_target: Union[list[Sample], list[Reaction]]
        ) -> Union[list[Sample], list[Reaction]]:
            l = []
            uids = sorted([s.uid for s in sort_target])
            for uid in uids:
                for target in sort_target:
                    if target.uid == uid:
                        l.append(target)

            return l

        if target == "samples" or target == "both":
            self.samples = _sort(self.samples)

        if target == "reactions" or target == "both":
            self.reactions = _sort(self.reactions)

    def get_dataloader(self, cfg: Optional[dict] = {}) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over dataset object.

        Args:
            cfg (dict, optional): Optional configuration for dataloader settings.

        Returns:
            DataLoader: Pytorch dataloader
        """

        def collate_fn(batch) -> tuple[list[Sample], list[Reaction]]:
            # NOTE: for first setup simply take in list of (samples, reaction) tuples
            # TODO: this does not parallelize well, for correct handling, tensorwise
            #       concatenation of samples and reactions properties is necessary

            # fixed number of partners
            # assert all([len(s[0]) == len(batch[0][0]) for s in batch])

            # TODO: add typehints for batch: List[List[List[Sample], Reaction]]
            def pad_batch(batch):
                """Pad batch to constant number of partners per reaction."""

                # max number of partners in batch
                max_partner = max([len(s[0]) for s in batch])

                # empty sample object used for padding
                ref = batch[0][0][0]
                padding_sample = Sample(
                    buid=ref.buid,
                    uid="PADDING",
                    numbers=torch.zeros_like(ref.numbers),
                    positions=torch.zeros_like(ref.positions),
                    unpaired_e=torch.zeros_like(ref.unpaired_e),
                    charges=torch.zeros_like(ref.charges),
                    egfn1=torch.zeros_like(ref.egfn1),
                    ggfn1=torch.zeros_like(ref.ggfn1),
                    egfn2=torch.zeros_like(ref.egfn2),
                    ggfn2=torch.zeros_like(ref.ggfn2),
                    eref=torch.zeros_like(ref.eref),
                    gref=torch.zeros_like(ref.gref),
                    edisp=torch.zeros_like(ref.edisp),
                    erep=torch.zeros_like(ref.erep),
                    qat=torch.zeros_like(ref.qat),
                    cn=torch.zeros_like(ref.cn),
                    ovlp=torch.zeros_like(ref.ovlp),
                    h0=torch.zeros_like(ref.h0),
                    adj=torch.zeros_like(ref.adj),
                )

                # pad each batch
                for s in batch:
                    for _ in range(max_partner - len(s[0])):
                        s[0].append(padding_sample)

                return batch

            # pad batch to same length
            batch = pad_batch(batch)

            batched_samples = [{} for _ in range(len(batch[0][0]))]
            batched_reactions = []

            for i, s in enumerate(batch):
                # assume standardised features
                assert all(
                    sample.positions.shape == s[0][0].positions.shape for sample in s[0]
                )
                assert all(
                    sample.numbers.shape == s[0][0].numbers.shape for sample in s[0]
                )
                assert all(sample.ovlp.shape == s[0][0].ovlp.shape for sample in s[0])
                assert all(sample.h0.shape == s[0][0].h0.shape for sample in s[0])

                # batch samples
                for j, sample in enumerate(s[0]):
                    if i == 0:
                        batched_samples[j] = sample.to_dict()
                        batched_samples[j]["uid"] = f"BATCH {j}"
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

            # stack features together
            partners = [i for r in batched_reactions for i in r.partners]
            nu = padded_stack(
                tensors=[r.nu for r in batched_reactions],
                side="right",
                value=0,  # padding value for stoichiometry factor
            )
            egfn1 = torch.stack([r.egfn1 for r in batched_reactions], 0)
            egfn2 = torch.stack([r.egfn2 for r in batched_reactions], 0)
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
                uid="BATCH",
                partners=partners,
                nu=nu,
                egfn1=egfn1,
                egfn2=egfn2,
                eref=eref,
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
        ) -> list[Union[None, tuple[int, int]]]:
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

    def prune(self) -> None:
        """Remove samples from the dataset that are not contained in any reaction."""
        # self.samples = [s for s in self.samples if len(s.partners) > 0]
        samples_new = []
        for s in self.samples:
            # check whether needed in any reaction
            for r in self.reactions:
                if s.uid.split(":")[1] in r.partners:
                    samples_new.append(s)
                    continue
        self.samples = samples_new

    def to_df(
        self, flatten: bool = True, path: Union[None, str] = None
    ) -> pd.DataFrame:
        """Convert dataset to pandas dataframe format.

        Parameters
        ----------
        flatten : bool, optional
            Flatten multidimensional features to scalar values, by default True.
        path : Union[None, str], optional
            Saving the dataframe to path on disk, by default None.

        Returns
        -------
        pd.DataFrame
            Dataset in dataframe format
        """

        # single padded batch
        loader = self.get_dataloader({"batch_size": len(self), "num_workers": 1})

        # data is "([Sample(BATCH 0), Sample(BATCH 1)], Reaction(BATCH))"
        data = next(iter(loader))
        samples, reaction = data

        d = {"subset": get_subsets_from_batched_reaction(reaction)}

        # reactions
        skip = ["__", "device", "dtype", "uid", "partners", "nu", "egfn2"]
        for slot in reaction.__slots__:
            if not any(sl in slot for sl in skip):
                d[f"r_{slot}"] = getattr(reaction, slot)

        # samples
        feat_skip = ["unpaired_e", "charges", "egfn2", "numbers", "positions", "cn"]
        skip = ["__", "device", "dtype", "uid", "qat"] + feat_skip
        for i, s in enumerate(samples):

            # FIXME: Cannot get nu this way, because uid is overwritten
            # for reaction in self.reactions:
            # if s.uid in reaction.partners:
            # for j, partner in enumerate(reaction.partners):
            # if partner == s.uid:
            # nu = reaction.nu[j]

            for slot in s.__slots__:
                if not any(sl in slot for sl in skip):
                    attr = getattr(s, slot)
                    print(slot, attr.shape)
                    if isinstance(attr, Tensor) and flatten:
                        # simply add all entries together

                        if slot == "h0":
                            d[f"s{i}_{slot}_max"] = torch.linalg.matrix_norm(
                                attr, ord=1
                            )
                            d[f"s{i}_{slot}_nuc"] = torch.linalg.matrix_norm(
                                attr, ord="nuc"
                            )
                            d[f"s{i}_{slot}_sing"] = torch.linalg.matrix_norm(
                                attr, ord=2
                            )
                            d[f"s{i}_{slot}_frob"] = torch.linalg.matrix_norm(attr)

                            d[f"s{i}_{slot}_det"] = torch.linalg.det(attr)

                            # very slow
                            # eigvals, _ = torch.lobpcg(attr)
                            # d[f"s{i}_{slot}_eig"] = eigvals.squeeze(-1)

                            d[f"s{i}_{slot}_eig"] = largest_eigenvalue(attr)

                        elif slot == "ovlp":
                            d[f"s{i}_{slot}_max"] = torch.linalg.matrix_norm(
                                attr, ord=1
                            )
                            d[f"s{i}_{slot}_nuc"] = torch.linalg.matrix_norm(
                                attr, ord="nuc"
                            )
                            d[f"s{i}_{slot}_sing"] = torch.linalg.matrix_norm(
                                attr, ord=2
                            )
                            d[f"s{i}_{slot}_frob"] = torch.linalg.matrix_norm(attr)

                            d[f"s{i}_{slot}_eig"] = largest_eigenvalue(attr)

                        else:
                            while len(attr.shape) > 1:
                                attr = torch.sum(attr, -1)
                            # TODO: find better ways to agglomerate vectors and matricies (sum, max, average, determinant, ..)
                            d[f"s{i}_{slot}"] = attr

        # TODO: multiply with stoichiometry factors

        # convert to dataframe
        df = pd.DataFrame(d)

        # save to disk
        if path:
            df.to_csv(path)

        return df


def store_subsets_on_disk(
    dataset: ReactionDataset,
    path: Union[Path, str],
    subsets: list[str],
):
    """Store subsets on disk by first pruning the dataset and then saving to disk.

    Parameters
    ----------
    dataset : ReactionDataset
        Dataset to be stored on disk
    path : Union[Path, str]
        Path to folder where subsets are stored (creates a reactions.json and samples.json file)
    subsets: List[str]
        List of subset names to be kept
    """
    # NOTE: this modifies the dataset in place!

    keep = [
        i for i, r in enumerate(dataset.reactions) if r.uid.split("_")[0] in subsets
    ]
    idxs = [i for i in range(len(dataset)) if i not in keep]

    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)

    # remove unneeded samples
    dataset.prune()

    # store subsets on disk
    dataset.to_json(path)


def get_dataset(
    path_reactions: Union[Path, str], path_samples: Union[Path, str]
) -> ReactionDataset:
    """Return a preliminary dataset for setting up the workflow."""

    # load json from disk
    dataset = ReactionDataset.from_json(
        path_reactions=path_reactions, path_samples=path_samples
    )

    # apply simple padding
    dataset.pad()

    return dataset


def get_gmtkn55_dataset(path: Path) -> ReactionDataset:
    """Return total GMTKN55 dataset.

    Parameters
    ----------
    path : Path
        Directory where JSON files are stored.

    Returns
    -------
    ReactionDataset
    """

    dataset = get_dataset(
        path_reactions=Path(path, "reactions_ACONF.json"),
        path_samples=Path(path, "samples_ACONF.json"),
    )

    # assert len(dataset) == 1505
    return dataset


# https://github.com/Thinklab-SJTU/ThinkMatch/issues/18
# https://en.wikipedia.org/wiki/Power_iteration
def largest_eigenvalue(matrix, n_iter=100):
    """Calculate largest eigenvalue of a matrix with power iteration.

    Parameters
    ----------
    matrix : torch.Tensor
        Matrix of shape (n, n)
    n_iter : int, optional
        Number of iterations, by default 100

    Returns
    -------
    float
        Largest eigenvalue
    """

    eigvals = torch.zeros(matrix.shape[0])

    for i in range(matrix.shape[0]):
        m = matrix[i]

        # initialize
        eigenvector = torch.rand(m.shape[0])

        for _ in range(n_iter):
            last_eigenvector = eigenvector
            eigenvector = m @ eigenvector
            eigenvector = eigenvector / torch.norm(eigenvector)

            if torch.norm(eigenvector - last_eigenvector) < 1e-4:
                break

        eigval = torch.dot(eigenvector, m @ eigenvector)
        eigvals[i] = eigval

        # eigval1 = torch.linalg.eigh(m)[0][0]
        # print(eigval, eigval1)

    return eigvals


def create_subset(dataset: ReactionDataset, keys: Union[str, list[str]]):
    """Prune given dataset to subsets indicated by keys. Note: changing the dataset in-place

    Args:
        dataset (ReactionDataset): Dataset to be reduced to subsets
        keys (Union[str, List[str]]): List of keys indicating the wanted subsets
    """

    # FIXME: during writing and reading this to list changes the shape of single values e.g. "G21IP h"

    subset_dict = {
        "barrier": [
            "BH76",
            "BHPERI",
            "BHDIV10",
            "INV24",
            "BHROT27",
            "PX13",
            "WCPT18",
        ],
        "thermo_small": [
            "W4-11",
            "G21EA",
            "G21IP",
            "DIPCS10",
            "PA26",
            "SIE4x4",
            "ALKBDE10",
            "YBDE18",
            "AL2X6",
            "HEAVYSB11",
            "NBPRC",
            "ALK8",
            "RC21",
            "G2RC",
            "BH76RC",
            "FH51",
            "TAUT15",
            "DC13",
        ],
        "thermo_large": [
            "MB16-43",
            "DARC",
            "RSE43",
            "BSR36",
            "CDIE20",
            "ISO34",
            "ISOL24",
            "C60ISO",
            "PArel",
        ],
        "nci_inter": [
            "RG18",
            "ADIM6",
            "S22",
            "S66",
            "HEAVY28",
            "WATER27",
            "CARBHB12",
            "PNICO23",
            "HAL59",
            "AHB21",
            "CHB6",
            "IL16",
        ],
        "nci_intra": [
            "IDISP",
            "ICONF",
            "ACONF",
            "Amino20x4",
            "PCONF21",
            "MCONF",
            "SCONF",
            "UPU23",
            "BUT14DIOL",
        ],
    }

    if isinstance(keys, str):
        keys = [keys]

    subsets = [subset for k in keys for subset in subset_dict[k]]

    keep = [
        i for i, r in enumerate(dataset.reactions) if r.uid.split("_")[0] in subsets
    ]
    idxs = [i for i in range(len(dataset)) if i not in keep]

    # TODO: better use torch.utils.data.Subset
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)
    return
