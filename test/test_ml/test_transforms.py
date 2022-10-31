import pytest
import torch
from torch_geometric.data import Data as pygData

from xtbml.ml.transforms import Pad_Hamiltonian

from .gmtkn55 import GMTKN55


class TestTransforms:
    """Testing custom transformations acting on data objects."""

    def test_pad_hamiltonian(self):

        transform = Pad_Hamiltonian(n_shells=10)

        # dummy matrix as substitute for hamiltonian
        h0 = torch.arange(64).reshape(8, 8)
        ovlp = torch.arange(64).reshape(8, 8)

        data = pygData(
            x=None,
            edge_index=None,
            h0=h0,
            ovlp=ovlp,
        )

        data_transformed = transform(data)

        assert data_transformed.h0.shape == torch.Size([10, 10])
        assert data_transformed.ovlp.shape == torch.Size([10, 10])
