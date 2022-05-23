""" Simple pytoch ML model for training purposes. """
from typing import Dict, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from egnn_pytorch import EGNN, EGNN_Network
import sys

from ..data.samples import Sample


class Simple_Net(nn.Module):
    """Simple NN to test functionality of learning based on Egfn1."""

    def __init__(self, simplicity):
        super().__init__()

        simplicity = 3

        self.simplicity = simplicity
        # self.simplicity = kwargs.get('simplicity',1)
        # O: Eref
        # 1: Egfn1
        # 2: Egfn1 sum atomic + MLP |
        # 3: Egfn1 atomic + MLP |
        # 4: atomic Egfn1 + MLP

        # TODO:
        # * implement simplicity 1
        # * implement simplicity 2
        # * implement simplicity 3

        if self.simplicity == 2:
            self.input = 1
            self.fc = nn.Linear(self.input, 1)
        elif self.simplicity == 3:
            self.input = 2
            self.fc = nn.Linear(self.input, 1)
            self.fc1 = nn.Linear(self.input, 5)
            self.fc2 = nn.Linear(5, 1)

        self.hidden = 5
        self.output = 1

        # self.fc1 = nn.Linear(self.input, self.hidden)
        # self.fc2 = nn.Linear(self.hidden, self.output)

    def forward(self, batched_samples: List, batched_reaction: List) -> Tensor:
        """Forward pass of model, i.e. prediction on single reaction and can be batched along first dimension.

        Args:
            batched_samples (List): Batch of samples.
            batched_reaction (List): Batch of reactions.

        Returns:
            Tensor: Forward pass.
        """

        # TODO: write tests for different simplicity
        # - 0:  Eref --> L = 0.0
        # - 1:  Egfn1_reaction --> L = L_gmtkn55
        #   2:

        if self.simplicity == 0:
            x = torch.clone(batched_reaction.eref)
            x.requires_grad = True
            return x
        elif self.simplicity == 1:
            x = torch.clone(batched_reaction.egfn1)
            x.requires_grad = True
            return x

        for i, reactant in enumerate(batched_samples):

            if self.simplicity == 2:
                egfn1 = torch.sum(reactant.egfn1, 1)
                x = egfn1.view(egfn1.shape[0], 1)
                x = F.leaky_relu(self.fc(x))

            elif self.simplicity == 3:
                egfn1 = torch.sum(reactant.egfn1, 1)
                eref = torch.clone(batched_reaction.eref).view(egfn1.shape[0], 1)
                x = egfn1.view(egfn1.shape[0], 1)
                x = torch.cat((x, eref), 1)
                x = F.leaky_relu(self.fc(x))
                # x = F.leaky_relu(self.fc1(x))
                # x = F.leaky_relu(self.fc2(x))

                # TODO: what happens if you give the MLP the Egfn1_s and the Eref_r?
                #   --> then the NN should actually just refer to the E_ref and all other weight go to zero

            if i == 0:
                reactant_contributions = x
            else:
                reactant_contributions = torch.cat((reactant_contributions, x), 1)

        # print(
        #    "reactant_contributions:",
        #    reactant_contributions.shape,
        #    reactant_contributions,
        # )

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)

        return result


class Basic_CNN(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()

        self.hidden_size = 3  # TODO: set as argument (bs/number of atomic features)
        self.kernel_size = 2  # dummy value

        self.input = 44  # 2305  # 55697  # TODO: set as argument
        self.hidden = 20  # TODO: make this a flexible parameter
        self.output = 1

        self.input_atm = 20  # TODO: set as argument

        # NOTE: as of now, reuse same CNN for H and S
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )

        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool3d((None, None, 1))
        self.flattening = False  # empirically False (pooling) is better

        self.fc_atm = nn.Linear(self.input_atm, self.hidden)
        # TODO: set as argument

        self.fc1 = nn.Linear(self.input, self.hidden * 2)
        self.fc2 = nn.Linear(self.hidden * 2, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.output)

    def forward(self, batched_samples: List, batched_reaction: List) -> Tensor:
        """Forward pass of model, i.e. prediction on single reaction and can be batched along first dimension.

        Args:
            batched_samples (List): Batch of samples.
            batched_reaction (List): Batch of reactions.

        Returns:
            Tensor: Forward pass.
        """

        # TODO: add typing List[Samples] List[Reactions]
        # INFO: len(batched_samples) == how many reactants take part in each reaction
        # INFO: len(batched_reactions) == how many reactions are there (== batch_size)

        # single reactant contribution
        for i, reactant in enumerate(batched_samples):
            # NOTE: apply same CNN on each reactant

            # overlap
            o = torch.unsqueeze(reactant.ovlp, 1)
            x1 = self.pool(F.leaky_relu(self.conv1(o)))
            x1 = F.leaky_relu(self.conv2(x1))

            # hamiltonian
            h = torch.unsqueeze(reactant.h0, 1)
            x2 = self.pool(F.leaky_relu(self.conv1(h)))
            x2 = F.leaky_relu(self.conv2(x2))
            # x2 = F.leaky_relu(self.conv2(x2))
            # print("o, h: ", x.shape, x2.shape)
            # print(reactant.ovlp.shape, reactant.h0.shape)

            # TODO: maybe use these as different channels for the CNN

            # merge features
            x = torch.cat((x1, x2), 1)

            # print(x.shape)
            if self.flattening:
                # flatten all dimensions except batch
                x = torch.flatten(x, start_dim=2)
                # torch.Size([2, 6, 576]) torch.Size([2, 6, 20])
                # torch.Size([2, 6, 596])
            else:
                # pooling
                x = self.avgpool(x)
                x = x.view(x.size(0), x.size(1), -1)
                # torch.Size([2, 6, 24]) torch.Size([2, 6, 20])
                # torch.Size([2, 6, 44])
                # NOTE: higher variance than flattening, peak results better(?)

            # atomwise features
            x_atm = torch.cat(
                (
                    torch.unsqueeze(reactant.cn, 1),
                    torch.unsqueeze(reactant.egfn1, 1),
                    torch.unsqueeze(reactant.edisp, 1),
                    torch.unsqueeze(reactant.erep, 1),
                    torch.unsqueeze(reactant.ees, 1),
                    torch.unsqueeze(reactant.qat, 1),
                ),
                1,
            )  # (bs, n_f, X)
            # print(x_atm)
            # print(x_atm.shape)

            # TODO: not used yet
            # print(reactant.charges)
            # print(reactant.unpaired_e)

            # NOTE: only non-zero elements are used (no-padding)
            # NOTE: requires CNNs
            """
            padding_idx = reactant.numbers.nonzero(as_tuple=True)[1][-1] + 1
            cn = torch.unsqueeze(reactant.cn[:, :padding_idx], 1)         
            """
            # TODO: remove

            # encoding of atomwise features
            x_atm = F.leaky_relu(self.fc_atm(x_atm))
            # NOTE: due to permutational invariance of atoms MLP preferred over CNN

            # merge
            # print(x.shape, x_atm.shape)
            x = torch.cat((x, x_atm), -1)
            # print(x.shape)

            # combined feature evaluation
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = self.fc3(x)

            # weighting by stoichiometry factor
            x = x.view(x.size(0), -1) * batched_reaction.nu[:, i].reshape(-1, 1)
            # print(x.shape) # (bs, 1)
            # TODO: check weighting applied correctly

            # store reactant contributions
            if i == 0:
                reactant_contributions = x
            else:
                reactant_contributions = torch.cat((reactant_contributions, x), 1)

        """print(
            "reactant_contributions:",
            reactant_contributions.shape,
            reactant_contributions,
        )"""

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)

        return result


class Basic_EGNN(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()

        self.egnn = EGNN_Network(
            num_tokens=21, dim=32, depth=3, only_sparse_neighbors=True
        )

        self.layer1 = EGNN(dim=2)
        self.layer2 = EGNN(dim=2)

        self.hidden_size = 2  # dummy value
        self.kernel_size = 2  # dummy value

        self.input = 14087  # TODO: set as argument
        self.hidden = 30
        self.output = 1

        # NOTE: as of now, reuse same CNN for all features
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(self.input, self.hidden * 2)
        self.fc2 = nn.Linear(self.hidden * 2, self.hidden)
        self.fc3 = nn.Linear(self.hidden, self.output)

    def get_adjacency(self, sample: Sample) -> Tensor:
        """Returns the adjacency matrix for the given sample."""

        # fixed distance cutoff
        cutoff = 3
        distances = torch.cdist(sample.xyz, sample.xyz, p=2)
        adj_mat = torch.where(
            distances < cutoff,
            True,
            False,
        )
        # TODO fix adjacency by coordination number cutoff
        return adj_mat

    def forward(self, batched_samples: List, batched_reaction: List) -> Tensor:
        # prediction on single reaction (can be batched along first dimension)

        # TODO: add typing List[Samples] List[Reactions]
        # INFO: len(batched_samples) == how many reactants take part in each reaction
        # INFO: len(batched_reactions) == how many reactions are there (== batch_size)

        # single reactant contribution
        for i, reactant in enumerate(batched_samples):
            # NOTE: apply same EGNN on each reactant

            # calculate adjacency matrix
            adj_mat = self.get_adjacency(reactant)  # TODO

            n_atms = reactant.xyz.shape[1]  # TODO: remove
            bs = adj_mat.shape[0]  # batch size

            # merge atomic features # TODO
            cn = torch.unsqueeze(reactant.cn, 2)
            edisp = torch.unsqueeze(reactant.edisp, 2)
            feats = torch.cat((cn, edisp), 2)
            # print(feats.shape)

            # EGNN_Network
            if False:
                # TODO: how to give the number of features?
                #       so far expecting only single value
                # TODO: num_tokens seems related to feats space --> Why?
                # --> for the time being maybe better use the simpler EGNN module (singular layer)
                # TODO: how to incorporate adj into EGNN simple layers?

                """feats = torch.randint(0, 21, (bs, n_atms))  # features
                feats_out, coords_out = self.egnn(
                    feats, reactant.xyz, adj_mat=adj_mat
                )  # (1, n_atms, 32), (1, n_atms, 3)

                print(feats_out.shape, coords_out.shape)"""

            # feats = torch.randn(bs, n_atms, 512)  # (bs, n_atms, n_feats)
            """feats, coords = self.layer1(feats, reactant.xyz)
            atm, coords = self.layer2(feats, coords)"""

            atm, coords = self.layer1(feats, reactant.xyz)

            # print(feats.shape, coords.shape)  # (bs, n_atms, n_feats), (bs, n_atms, 3)
            import sys

            # sys.exit(0)

            # overlap
            o = torch.unsqueeze(reactant.ovlp, 1)
            o = self.pool(F.leaky_relu(self.conv1(o)))
            o = self.pool(F.leaky_relu(self.conv2(o)))

            # hamiltonian
            h = torch.unsqueeze(reactant.h0, 1)
            h = self.pool(F.leaky_relu(self.conv1(h)))
            h = self.pool(F.leaky_relu(self.conv2(h)))

            # merge features
            x = torch.cat((o, h), 1)
            x = torch.flatten(x, 1)  # flatten all dimensions except batch
            atm = torch.flatten(atm, 1)
            x = torch.cat((x, atm), 1)
            # x = torch.cat((x, cn, edisp), 1)
            # NOTE: repulsion is taken out due to too large values
            # x = torch.cat((x, cn, erep, edisp), 1)

            # add GFN1-xtb energy
            e = torch.unsqueeze(reactant.egfn1, 1)  # TODO: check unsqueeze dimension
            x = torch.cat((x, e), 1)

            if False:
                import sys

                print("VERBOSE END")
                sys.exit(0)

            # combined feature evaluation
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = self.fc3(x)

            # weighting by stoichiometry factor
            x = x * batched_reaction.nu[:, i].reshape(-1, 1)

            # store reactant contributions
            if i == 0:
                reactant_contributions = x
            else:
                reactant_contributions = torch.cat((reactant_contributions, x), 1)

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)

        return result
