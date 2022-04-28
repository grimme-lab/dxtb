from typing import Dict, List, Optional, Union, Tuple
from pydantic import BaseModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch
from torch import Tensor

# from sample import Sample
# from reaction import Reaction

##########################################################################################
# TODO: for reference only
class Sample(BaseModel):
    """Representation for single sample information."""

    # supported features
    uid: str
    """Unique identifier for sample"""
    positions: Tensor
    """Atomic positions"""
    atomic_numbers: Tensor
    """Atomic numbers"""
    egfn1: Tensor
    """Energy calculated by GFN1-xtb"""
    overlap: Tensor
    """Overlap matrix"""
    hamiltonian: Tensor
    """Hamiltonian matrix"""

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True


# TODO: for reference only
class Reaction(BaseModel):
    """Representation for reaction involving multiple samples."""

    uid: str
    """Unique identifier for reaction"""
    reactants: List[str]  # reactants, participants, partner, ...
    """List of reactants uids"""
    nu: Union[List[int], Tensor]
    """Stoichiometry coefficient for respective participant"""
    egfn1: Tensor
    """Reaction energies given by GFN1-xtb"""
    eref: Tensor
    """Reaction energies given by reference method"""

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True


##########################################################################################


class Reaction_Dataset(BaseModel, Dataset):
    """Dataset for storing features used for training."""

    # TODO: better would be an object of lists than a list of objects
    samples: List[Sample]
    """Samples in dataset"""
    reactions: List[Reaction]
    """Reactions in dataset"""

    def __len__(self):
        """Length of dataset defined by number of reactions."""
        return len(self.reactions)

    def __getitem__(self, idx: int):
        """Get all samples involved in specified reaction."""
        reaction = self.reactions[idx]
        samples = [s for s in self.samples if s.uid in reaction.reactants]
        return samples, reaction

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

            # fixed number of reactants
            assert all([len(s[0]) == len(batch[0][0]) for s in batch])

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
                    print(sample.uid)
                    if i == 0:
                        batched_samples[j] = sample.dict()
                        print("here we are")
                        for k, v in batched_samples[j].items():
                            if isinstance(v, Tensor):
                                batched_samples[j][k] = v.unsqueeze(0)
                                print(k, v.shape)
                        continue

                    for k, v in sample.dict().items():
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
            reactants = [i for r in batched_reactions for i in r.reactants]
            nu = torch.tensor([r.nu for r in batched_reactions])
            egfn1 = torch.stack([r.egfn1 for r in batched_reactions], 0)
            eref = torch.stack([r.eref for r in batched_reactions], 0)

            # NOTE: Example on how batching of Reaction objects is conducteds
            # [Reaction(uid='AB', reactants=['A', 'B', 'AB'], nu=[-1, -1, 1],
            # egfn1=tensor([1.2300]), eref=tensor([1.5400])),
            # Reaction(uid='AC', reactants=['A', 'C', 'AC'], nu=[-1, -1, 1],
            # egfn1=tensor([3.4500]), eref=tensor([7.2300]))]
            # -->
            # Reaction(uid='BATCH', reactants=[['A', 'B', 'AB'], ['A', 'C', 'AC']], nu=[[-1, -1, 1], [-1, -1, 1]],
            # egfn1=tensor([[1.2300], [3.4500]]), eref=tensor([[1.5400], [7.2300]])),

            # convert to sample objects (optional)
            batched_samples = [Sample(**d) for d in batched_samples]
            batched_reactions = Reaction(
                uid="BATCH", reactants=reactants, nu=nu, egfn1=egfn1, eref=eref
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
