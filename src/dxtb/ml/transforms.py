"""Module containing transformations applicable to datasets."""
import torch
from torch import nn


class Pad_Hamiltonian(nn.Module):
    def __init__(self, n_shells: int) -> None:
        """Padding of shell-resolved quantities such as hamiltonian and overlap.
            Fixes shape to this value -- shorter tensors get padded, larger pruned.

        Args:
            n_shells (int): Number of shells to be padded to
        """
        self.n_shells = n_shells

    def __call__(self, data):
        """Pad data tensors

        Args:
            data (torch_geometric.data.data.Data): Single graph data sample

        Returns:
            torch_geometric.data.data.Data: Padded graph data sample
        """

        diff = self.n_shells - data.h0.shape[0]

        # simple zero padding in one direction
        pad = (0, diff, 0, diff)
        h0 = torch.nn.functional.pad(data.h0, pad, "constant", 0)
        ovlp = torch.nn.functional.pad(data.ovlp, pad, "constant", 0)

        # update sample
        data.h0 = h0
        data.ovlp = ovlp
        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
