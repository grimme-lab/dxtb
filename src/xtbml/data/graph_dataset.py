import sys
import torch
from torch_geometric.data import InMemoryDataset, Data as pygData
from pathlib import Path

from .adjacency import calc_adj
from .samples import Sample
from ..typing import Tensor
from .dataset import get_gmtkn55_dataset, SampleDataset


class MolecularGraph_Dataset_Parametrisation:
    """Define parametrisation for different benchmark datasets."""

    def __init__(self, root: Path):
        """
        Basic implementation for an object holding the information on how to load different benchmarks into the MolecularGraph_Dataset.
        Hereby, processed_file_names indicates the name of the (processed) files on disk. If those are present, dataloading is skipped.
        Note that the get_dataset interface requires to return a List[Sample].

        Parameters
        ----------
        root : Path
            Path to folder containing benchmark data. Should end with specification of dataset.

        Raises
        ------
        KeyError
            Unknown dataset to be called. Check constructor for definition and detection of implemented datasets.
        """

        # NOTE: define adapter to dataset here, but call later
        if root.name == "GMTKN55":
            self.get_dataset = get_gmtkn55_dataset
            self.processed_file_names = ["gmtkn55.pt"]

        elif root.name == "PTB":

            def get_ptb_dataset(root: Path) -> SampleDataset:
                list_of_path = sorted(root.glob("samples_*.json"))
                return SampleDataset.from_json(list_of_path)

            self.get_dataset = get_ptb_dataset
            self.processed_file_names = ["ptb.pt"]
        else:
            raise KeyError(f"Unknown dataset for MolecularGraph_Dataset: {root}")


class MolecularGraph_Dataset(InMemoryDataset):
    def __init__(
        self,
        root: Path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        # parametrisation for different benchmark datasets
        self.data_param = MolecularGraph_Dataset_Parametrisation(root)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return self.data_param.processed_file_names

    def process(self):
        """Processing data. Currently loading GMTKN-55 via dataset and converting into pytorch-geometric graph objects."""

        # load dataset via adapter
        dataset = self.data_param.get_dataset(self.root)

        # extract all samples from reactions and convert to graph objects
        data_list = [self.convert_to_graph(sample) for sample in dataset.samples]
        assert all([isinstance(s, pygData) for s in data_list])

        # apply filter and pre-transformations
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # collate graphs into single objects
        data, slices = self.collate(data_list)

        # save processed data to disk
        torch.save((data, slices), self.processed_paths[0])

    def convert_to_graph(self, sample: Sample) -> pygData:
        """Convert a given sample to graph representation given in a pytorch-geometric pygData object.

        Args:
            sample (Sample): Sample containing molecular information.

        Returns:
            pygData: Graph representation of given sample.
        """

        print("convert_to_graph: ", sample)

        # calculate adjacency matricies
        if sample.adj.nelement() == 0:
            sample.adj = calc_adj(sample).type(sample.dtype)

        edge_index = self._get_edge_index(sample.adj)  # [2, number of edges]

        edge_attr = self._get_edge_attr(
            edge_index.size(1)
        )  # [number of edges, number of edge attributes]

        node_attr = self._get_node_attr(
            sample
        )  # [number of nodes, number of node features]

        # atomic positions for e.g. equivariant calculations
        positions = sample.positions  # [number of nodes, 3]

        # graph-wide target
        target = torch.sum(sample.egfn1).view([1, 1])  # [1, *]

        # sample and dataset name
        uid = sample.uid
        buid = sample.buid
        # dummy
        egfn1 = torch.tensor(123.4)  # TODO
        egfn2 = torch.tensor(666.6)

        return pygData(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            y=target,
            uid=uid,
            buid=buid,
            egfn1=egfn1,
            egfn2=egfn2,
        )

    def _get_edge_index(self, adjacency_matrix: Tensor) -> Tensor:
        """Calculate edge indices from adjacency matrix.

        Args:
            adjacency_matrix (Tensor): Symmetric binary matrix containing information on neighboring atoms.

        Returns:
            Tensor: Edge indices with shape [2, number of edges]
        """
        return adjacency_matrix.nonzero().t().contiguous()

    def _get_edge_attr(self, num_edges: int) -> Tensor:

        # TODO: add Hamiltonian and Overlap as edge weights
        #       (use index helper), also add diagonal entries
        #       to node attributes

        # NOTE: currently no edge attributes are added
        num_edge_features = 1

        return torch.ones(
            (num_edges, num_edge_features),
            dtype=torch.float,
        )  # [number of edges, number of edge attributes]

    def _get_node_attr(self, sample: Sample) -> Tensor:

        # NOTE: so far only atom-wise features
        return torch.stack(
            [
                sample.egfn1,
                sample.edisp,
                sample.erep,
                sample.qat,
                sample.cn,
            ],
            dim=1,
        )  # [number of nodes, number of node features]
