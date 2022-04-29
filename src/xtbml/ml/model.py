""" Simple pytoch ML model for training purposes. """
from typing import Dict, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class Basic_CNN(nn.Module):
    def __init__(self, cfg: Dict[str, int]):
        super().__init__()

        self.hidden_size = 2  # dummy value
        self.kernel_size = 2  # dummy value

        self.input = 4357  # TODO: set as argument
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

    def forward(self, batched_samples: List, batched_reaction: List) -> Tensor:
        # prediction on single reaction (can be batched along first dimension)

        # INFO: len(batched_samples) == how many reactants take part in each reaction
        # INFO: len(batched_reactions) == how many reactions are there (== batch_size)

        # single reactant contribution
        for i, reactant in enumerate(batched_samples):
            # NOTE: apply same CNN on each reactant

            # overlap
            o = torch.unsqueeze(reactant.ovlp, 1)
            x = self.pool(F.leaky_relu(self.conv1(o)))
            x = F.leaky_relu(self.conv2(x))

            # hamiltonian
            h = torch.unsqueeze(reactant.h0, 1)
            x2 = self.pool(F.leaky_relu(self.conv1(h)))
            x2 = F.leaky_relu(self.conv2(x2))

            # merge features
            x = torch.cat((x, x2), 1)
            x = torch.flatten(x, 1)  # flatten all dimensions except batch

            # add GFN1-xtb energy
            e = torch.unsqueeze(reactant.egfn1, 1)  # TODO: check unsqueeze dimension
            x = torch.cat((x, e), 1)

            # combined feature evaluation
            x = F.leaky_relu(self.fc1(x))
            x = F.leaky_relu(self.fc2(x))
            x = self.fc3(x)

            # weighting by stoichiometry factor
            x = x * batched_reaction.nu[:, i].reshape(-1, 1)
            # print(x.shape) # (bs, 1)

            # store reactant contributions
            if i == 0:
                reactant_contributions = x
            else:
                reactant_contributions = torch.cat((reactant_contributions, x), 1)

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)

        return result
