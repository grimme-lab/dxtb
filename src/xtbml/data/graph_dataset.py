from __future__ import annotations

from pathlib import Path
from tkinter import N

import torch
from torch_geometric.data import Data as pygData
from torch_geometric.data import InMemoryDataset

from ..basis import IndexHelper, MatrixHelper, get_elem_param_shells
from ..param.gfn1 import GFN1_XTB as par
from ..typing import Tensor
from .adjacency import calc_adj
from .dataset import SampleDataset, get_gmtkn55_dataset
from .samples import Sample


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
                list_of_path = sorted(
                    root.glob("samples_*.json")
                )  # samples_*.json samples_HE*.json samples_DUMMY*.json
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
        # print("edge_index", edge_index.shape)

        edge_attr = self._get_edge_attr(
            edge_index.size(1), sample
        )  # [number of edges, number of edge attributes]
        # print("edge_attr", edge_attr.shape)

        node_attr = self._get_node_attr(
            sample
        )  # [number of nodes, number of node features]
        # print("node_attr", node_attr.shape)

        # atomic positions for e.g. equivariant calculations
        positions = sample.positions  # [number of nodes, 3]
        # print("positions", positions.shape)

        # graph-wide target
        target = torch.sum(sample.egfn1).view([1, 1])  # [1, *]
        # print("target", target.shape)

        return pygData(
            x=node_attr,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=positions,
            # y=target,
            # sample and dataset name
            uid=sample.uid,
            buid=sample.buid,
            # energies and gradients
            egfn1=sample.egfn1,
            egfn2=sample.egfn2,
            eref=sample.eref,
            ggfn1=sample.ggfn1,
            ggfn2=sample.ggfn2,
            gref=sample.gref,
            # molecular properties
            numbers=sample.numbers,
            charges=sample.charges,
            unpaired_e=sample.unpaired_e,
            # shell resolved
            h0=sample.h0,
            ovlp=sample.ovlp,
        )

    def _get_edge_index(self, adjacency_matrix: Tensor) -> Tensor:
        """Calculate edge indices from adjacency matrix.

        Args:
            adjacency_matrix (Tensor): Symmetric binary matrix containing information on neighboring atoms.

        Returns:
            Tensor: Edge indices with shape [2, number of edges]
        """
        return adjacency_matrix.nonzero().t().contiguous()

    def _get_edge_attr(self, num_edges: int, sample: Sample) -> Tensor:

        # NOTE: currently no edge attributes are added
        num_edge_features = 1
        edge_attr = torch.ones(
            (num_edges, num_edge_features),
            dtype=torch.float,
        )

        if False:
            # adding Hamiltonian blocks as edge attributes

            # anuglar momenta and respective orbitals for different elements
            angular, valence_shells = get_elem_param_shells(par.element, valence=True)

            ihelp = IndexHelper.from_numbers(sample.numbers, angular)

            n_atoms = sample.adj.shape[0]

            # get hamiltonian block for all atomic combinations
            combi = torch.combinations(
                torch.arange(n_atoms), r=2, with_replacement=True
            )
            for i, j in combi:
                block_ij = MatrixHelper.get_atomblock(sample.h0, i, j, ihelp)
                print(block_ij.shape)
            # TODO: how to add matricies of different size to pyg?
            #       (physically meaningful mapping to graph)

        return edge_attr  # [number of edges, number of edge attributes]

    def _get_node_attr(self, sample: Sample) -> Tensor:
        """Stack all relevant atom-wise features to single node tensor.
        Each node corresponds to single atom in molecule.

        Parameters
        ----------
        sample : Sample
            Sample containing information on single molecule

        Returns
        -------
        Tensor
            Node attributes of shape [number of nodes, number of node features]
        """

        # TODO (optional): hamiltonian spread on atoms
        # orb_sums = MatrixHelper.get_orbital_sum(sample.h0, ihelp)
        # atomic_sums = torch.stack([torch.sum(orb) for orb in orb_sums])
        # print(atomic_sums)

        return torch.stack(
            [
                sample.egfn1,
                sample.edisp,
                sample.erep,
                sample.qat,
                sample.cn,
            ],
            dim=1,
        )
