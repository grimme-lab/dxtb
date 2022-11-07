import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import Data, Batch
from typing import List, Dict


import sys


class GNN(torch.nn.Module):
    """Graph based model."""

    def __init__(self, cfg: Dict[str, int]):
        super().__init__()

        # TODO: wrap into config dictionary
        nf = 5
        out = 2

        self.gconv1 = GCNConv(nf, 50)
        self.gconv2 = GCNConv(50, out)

        # atomwise energies:
        self.energy_atm = torch.nn.Linear(out, 1)

    def forward(self, batched_samples: List, batched_reaction: List):
        verbose = False

        if verbose:
            print("inside forward")
        # x = ...           # Node features of shape [num_nodes, num_features]
        # edge_index = ...  # Edge indices of shape [2, num_edges] (type(torch.long))

        reactant_contributions = []
        for i, s in enumerate(batched_samples):
            if verbose:
                print(s, s.adj.shape)

            # for graph batching see
            #   https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
            #   https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
            #   https://github.com/pyg-team/pytorch_geometric/issues/973

            # node features
            batch_x = torch.stack(
                [
                    s.egfn1,
                    s.edisp,
                    s.erep,
                    s.qat,
                    s.cn,
                ],
                dim=2,
            )  # [batch size, number of nodes, number of node features]

            # TODO: add encoding of hamiltonian and overlap

            ######
            if False:
                print("before", batch_x.shape)

                # TODO: set as argument (bs/number of atomic features)
                self.hidden_size = 3
                self.kernel_size = 2  # dummy value

                # NOTE: as of now, reuse same CNN for H and S
                self.conv1 = torch.nn.Conv2d(
                    in_channels=1,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                )
                self.conv2 = torch.nn.Conv2d(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=0,
                )

                self.pool = torch.nn.MaxPool2d(2, 2)

                # overlap
                o = torch.unsqueeze(s.ovlp, 1)
                x1 = self.pool(F.leaky_relu(self.conv1(o)))
                x1 = F.leaky_relu(self.conv2(x1))

                # hamiltonian
                h = torch.unsqueeze(s.h0, 1)
                x2 = self.pool(F.leaky_relu(self.conv1(h)))
                x2 = F.leaky_relu(self.conv2(x2))

                print(x2.shape)

                x12 = torch.cat((x1, x2), 1)
                print(x12.shape)

                print("adding overlap and H0")
                sys.exit(0)
                # TODO: how to best add H0 info? is there a possibility to
                #       incooperate the shell info onto a graph level?
                #       so far we have discussed that this is not really
                #       possible however, how is this done in the orbnet?
            ######

            batch_size, num_nodes, num_in_node_features = batch_x.shape

            # wrap input nodes, edge index and weights into 'torch_geometric.data.Batch'
            graph_list = []
            for j in range(batch_size):
                edge_index = s.adj[j].nonzero().t().contiguous()  # [2, number of edges]
                num_in_edge_features = 1
                edge_attr = torch.ones(
                    (edge_index.size(1), num_in_edge_features),
                    dtype=torch.float,
                )  # [number of edges, number of edge attributes]
                # NOTE: currently no edge attributes are added
                # print("edge_attr", edge_attr.shape)

                graph_list.append(
                    Data(
                        x=batch_x[j],
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                    )
                )
            batch = Batch.from_data_list(graph_list)
            assert batch.num_graphs == batch_size
            # retrieving which node belongs to which graph
            if verbose:
                print(batch)
                print(batch.batch)

            if False:
                # dummy version - dynamic graphs (unpractically, new nn for each new graph)
                num_out_node_features = 64
                nn = torch.nn.Sequential(
                    torch.nn.Linear(
                        num_in_edge_features, 25
                    ),  # flexible due to dynamic batches
                    torch.nn.ReLU(),
                    torch.nn.Linear(25, num_in_node_features * num_out_node_features),
                )
                gconv = NNConv(
                    in_channels=num_in_node_features,
                    out_channels=num_out_node_features,
                    nn=nn,
                    aggr="mean",
                )
                y = gconv(
                    x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr
                )  # [num of nodes in batch, number of output features]

            if verbose:
                print("before", batch.x.shape)

            x = self.gconv1(batch.x, batch.edge_index)
            x = F.silu(x)
            # x = F.dropout(x, p=0.2, training=self.training)
            x = self.gconv2(x, batch.edge_index)
            # TODO: assert disjunct graphs, i.e. correct batching and no-cross inference

            if verbose:
                print("after", x.shape)

            # NOTE: currently infering per sample energy (physically motivated)
            ## Version 1 -- atomwise energies

            # split into seperate tensors by batch.batch index
            _, cnt = torch.unique(batch.batch, return_counts=True)

            # check for same size of graphs (can be removed -- due to padding)
            assert len(torch.unique(cnt)) == 1
            assert (cnt == num_nodes).all()

            idx = 0
            e_batch = torch.zeros([batch_size])
            for j, ll in enumerate(cnt):
                # select and calculate atomic energy per sample in batch
                x_i = x[idx : idx + ll, :]
                idx += ll.item()
                e_atm = self.energy_atm(x_i)  # [number of nodes, 1]
                # sum over atomic energies to get molecular energies
                e_batch[j] = torch.sum(e_atm, dim=0)  # [1]

            # weighting by stoichiometry factor
            e_batch = e_batch * batched_reaction.nu[:, i]  # [bs]

            ## Version 2 -- directly infer molecular energies
            # TODO: (C)NN [number of nodes, number of features] -> [1]

            # store reactant contributions
            reactant_contributions.append(e_batch.unsqueeze(1))

        if verbose:
            print(len(reactant_contributions))
            print([i.shape for i in reactant_contributions])
        reactant_contributions = torch.cat(
            reactant_contributions, dim=1
        )  # [bs, number of reactants]

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)  # [bs]

        return result
