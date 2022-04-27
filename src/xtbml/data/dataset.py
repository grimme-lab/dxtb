from typing import Dict, List
from pydantic import BaseModel
from torch.utils.data import Dataset

from sample import Sample
from reaction import Reaction


class Feature_Dataset(BaseModel, Dataset):
    """Dataset for storing features used for training."""

    # TODO: better would be an object of lists than a list of objects
    samples: List[Sample]
    """Samples in dataset"""
    reactions: List[Reaction]
    """Reactions in dataset"""

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx: int):
        # TODO: rather pseudo-code
        return [sample for sample in self.samples[idx]], self.reactions[idx]

    def create_from_disk(csv1: str, csv2: str) -> "Feature_Dataset":
        raise NotImplementedError
