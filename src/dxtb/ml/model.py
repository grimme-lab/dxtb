""" Simple pytoch ML model for testing purposes. """
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

class Simple_Net(nn.Module):
    """Simple NN to test functionality of learning based."""

    def __init__(self, simplicity):
        super().__init__()

        self.simplicity = simplicity
        # O: Eref
        # 1: Egfn1
        # 2: Egfn1 sum atomic + MLP
        # 3: Egfn1 atomic + MLP

        # required for non-empty parameter list
        self.dummy = nn.Linear(1, 1)
        if self.simplicity == 2:
            self.input = 1
            self.fc = nn.Linear(self.input, 1)


    def forward(self, batched_samples: list, batched_reaction: list) -> Tensor:
        """Forward pass of model, i.e. prediction on single reaction and can be batched along first dimension.

        Args:
            batched_samples (list): Batch of samples.
            batched_reaction (list): Batch of reactions.

        Returns:
            Tensor: Forward pass.
        """

        if self.simplicity == 0:
            x = torch.clone(batched_reaction.eref)
            x.requires_grad = True
            return x
        elif self.simplicity == 1:
            x = torch.clone(batched_reaction.egfn1)
            x.requires_grad = True
            return x

        for i, reactant in enumerate(batched_samples):

            egfn1 = torch.sum(reactant.egfn1, 1)
            x = egfn1.view(egfn1.shape[0], 1)
            x = F.leaky_relu(self.fc(x))

            if i == 0:
                reactant_contributions = x
            else:
                reactant_contributions = torch.cat((reactant_contributions, x), 1)

        # sum over reactant contributions
        result = torch.sum(reactant_contributions, 1)

        return result

