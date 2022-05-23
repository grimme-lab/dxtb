from typing import List, Optional, Union, Tuple, overload
from pydantic import BaseModel
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

from xtbml.data.reactions import Reaction, Reactions
from xtbml.data.samples import Sample, Samples


# TODO: add to general utils
# With courtesy to https://pytorch-forecasting.readthedocs.io
def padded_stack(
    tensors: List[torch.Tensor],
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

    def to_disk(self, path: Union[Path, str]):
        """Save dataset to disk.

        Args:
            path (str): Path to save dataset to.
        """
        Reactions(reactions=self.reactions).to_json(path)
        Samples(samples=self.samples).to_json(path)

    def get_samples_from_reaction_partners(self, reaction: Reaction) -> List[Sample]:
        sample_list = []
        for partner in reaction.partners:
            for sample in self.samples:
                if sample.uid == partner:
                    sample_list.append(sample)

        return sample_list

    def __len__(self):
        """Length of dataset defined by number of reactions."""
        return len(self.reactions)

    @overload
    def __getitem__(self, idx: int) -> Tuple[List[Sample], Reaction]:
        ...

    @overload
    def __getitem__(self, idx: slice) -> "ReactionDataset":
        ...

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Tuple[List[Sample], Reaction], "ReactionDataset"]:
        """Get all samples involved in specified reaction."""

        reactions = self.reactions[idx]

        if isinstance(idx, slice) and isinstance(reactions, list):
            samples = [
                self.get_samples_from_reaction_partners(reaction)
                for reaction in reactions
            ]

            return ReactionDataset(
                samples=list(set(item for sublist in samples for item in sublist)),
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
    @classmethod
    def merge(cls, a, b):
        """Merge two datasets."""
        return cls(samples=a.samples + b.samples, reactions=a.reactions + b.reactions)

    # FIXME: Not needed currently -> remove?
    def rm_reaction(self, idx: int):
        """Remove reaction from dataset."""
        # NOTE: Dataset might contain samples
        #       that are not required in any reaction anymore
        del self.reactions[idx]

    def equal(self, other):
        if not isinstance(other, ReactionDataset):
            # do not compare against unrelated types
            return NotImplemented

        for i, sample in enumerate(self.samples):
            if not sample.equal(other.samples[i]):
                return False

        for i, reaction in enumerate(self.reactions):
            if not reaction.equal(other.reactions[i]):
                return False

        return True

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

            # TODO: add typehints for batch: List[List[List[Sample], Reaction]]
            def pad_batch(batch):
                """Pad batch to constant number of partners per reaction."""

                # max number of partners in batch
                max_partner = max([len(s[0]) for s in batch])

                # empty sample object used for padding
                ref = batch[0][0][0]
                padding_sample = Sample(
                    uid="PADDING",
                    xyz=torch.zeros_like(ref.xyz),
                    numbers=torch.zeros_like(ref.numbers),
                    unpaired_e=torch.zeros_like(ref.unpaired_e),
                    charges=torch.zeros_like(ref.charges),
                    egfn1=torch.zeros_like(ref.egfn1),
                    egfn2=torch.zeros_like(ref.egfn2),
                    edisp=torch.zeros_like(ref.edisp),
                    erep=torch.zeros_like(ref.erep),
                    ovlp=torch.zeros_like(ref.ovlp),
                    h0=torch.zeros_like(ref.h0),
                    cn=torch.zeros_like(ref.cn),
                    ees=torch.zeros_like(ref.ees),
                    qat=torch.zeros_like(ref.qat),
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
                assert all(sample.xyz.shape == s[0][0].xyz.shape for sample in s[0])
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

    def prune(self) -> None:
        """Remove samples from the dataset that are not contained in any reaction."""
        # self.samples = [s for s in self.samples if len(s.partners) > 0]
        samples_new = []
        for s in self.samples:
            # check whether needed in any reaction
            for r in self.reactions:
                if s.uid in r.partners:
                    samples_new.append(s)
                    continue
        self.samples = samples_new


def store_subsets_on_disk(
    dataset: ReactionDataset,
    path: Union[Path, str],
    subsets: List[str],
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
    dataset.to_disk(path)


def get_dataset(
    path_reactions: Union[Path, str], path_samples: Union[Path, str]
) -> ReactionDataset:
    """Return a preliminary dataset for setting up the workflow."""

    # load json from disk
    dataset = ReactionDataset.create_from_disk(
        path_reactions=path_reactions, path_samples=path_samples
    )

    # apply simple padding
    dataset.pad()

    return dataset


def get_gmtkn_dataset(path: Path) -> ReactionDataset:
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
        path_reactions=Path(path, "reactions.json"),
        path_samples=Path(path, "samples.json"),
        # path_reactions=Path(path, "reactions_0.json"),  # ACONF
        # path_samples=Path(path, "samples_0.json"),
        # path_reactions=Path(path, "reactions_1.json"), # ACONF, Amino20x4, MCONF, PCONF21
        # path_samples=Path(path, "samples_1.json"),
        # path_reactions=Path(path, "reactions_2.json"),  # ADIM6, S22, S66
        # path_samples=Path(path, "samples_2.json"),
        # path_reactions=Path(path, "reactions_12.json"), # ACONF, Amino20x4, MCONF, PCONF21, ADIM6, S22, S66
        # path_samples=Path(path, "samples_12.json"),
    )

    assert len(dataset) == 1505
    return dataset
