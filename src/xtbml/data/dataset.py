from typing import Dict, List, Optional
from pydantic import BaseModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


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

    def padding(self, max_atoms: int) -> "Feature_Dataset":
        # TODO: add some padding, such that the samples features have same shape
        raise NotImplementedError

    def get_dataloader(self, cfg: Optional[dict] = {}) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over dataset object.

        Args:
            cfg (dict, optional): Optional configuration for dataloader settings.

        Returns:
            DataLoader: Pytorch dataloader
        """

        def collate_fn(batch):
            # TODO: add customised batching method if needed
            raise NotImplementedError
            batched_data = {}
            return batched_data

        if "collate_fn" not in cfg:
            cfg["collate_fn"] = collate_fn

        return DataLoader(self, **cfg)
