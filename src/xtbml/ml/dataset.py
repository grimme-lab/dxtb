from typing import Dict, Optional
from pydantic import BaseModel
import torch
from torch import Tensor
from torch.utils.data import DataLoader


""" Class for storing features used for training and inference. """
# TODO: same class for inference?


class Feature_Dataset(BaseModel):
    # alternative names: Sample_Dataset
    # optional: inherit from Geometry or torch.dataset

    # shape: (batch_size, max_atoms, ...)

    # supported features
    """Atomic positions"""  # TODO: possible to check for shape?
    positions: Tensor
    """Atomic numbers"""
    atomic_numbers: Tensor
    """Overlap matrix"""
    overlap: Tensor

    class Config:
        # allow for geometry and tensor fields
        arbitrary_types_allowed = True

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx: int) -> Dict:
        return {
            "atomic_numbers": self.atomic_numbers[idx],
            "positions": self.positions[idx],
            "overlap": self.overlap[idx],
        }

    # def __getitem__(self, idx: int) -> "Feature_Dataset":
    #    """Return slice of batched data."""
    #    return self.__class__(
    #        self.atomic_numbers[idx],
    #        self.positions[idx],
    #        self.overlap[idx],
    #    )  # TODO: difficult to combine with pydantic -- either leave this an return simple tensors or leave out pydantic

    def __str__(self) -> str:
        # only required when getitem returns a sliced object
        # if len(self.positions.shape) == 2:
        #    # single sample
        #    return super().__str__(self)  # TODO: call str method from super class

        return f"{self.__class__.__name__} containing {len(self)} samples"

    def get_dataloader(self, cfg: Optional[dict] = {}) -> DataLoader:
        """
        Return pytorch dataloader for batch-wise iteration over dataset object.

        Args:
            cfg (dict, optional): Optional configuration for dataloader settings.

        Returns:
            DataLoader: Pytorch dataloader
        """

        def collate_fn(batch):
            """Merge features along new first dimension."""
            batched_data = {}
            for s in batch:
                for k, v in s.items():
                    if k not in batched_data:
                        batched_data[k] = v.unsqueeze(0)
                    else:
                        batched_data[k] = torch.concat(
                            (batched_data[k], v.unsqueeze(0)), dim=0
                        )
            return batched_data

        if "collate_fn" not in cfg:
            cfg["collate_fn"] = collate_fn

        return DataLoader(self, **cfg)
