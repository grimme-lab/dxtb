""" Module for custom loss functions."""

from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List, Literal

from ..typing import Tensor
from ..data.dataset import get_gmtkn55_dataset


class WTMAD2Loss(torch.nn.Module):
    """Calculate the weighted total mean absolute deviation, as defined in

    - L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich, A. Najibi, Asim, S. Grimme,
      *Phys. Chem. Chem. Phys.*, **2017**, 19, 48, 32184-32215.
      (`DOI <http://dx.doi.org/10.1039/C7CP04913G>`__)
    """

    def __init__(
        self,
        path: Path,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        """_summary_

        Parameters
        ----------
        path : Path
            Absolute path of GMTKN55 directory
        reduction : str, optional
            Reduction of batch-wise loss to single value, by default "mean"

        Raises
        ------
        TypeError
            Invalid loss given
        """
        super(WTMAD2Loss, self).__init__()

        # calculate properties dynamically
        self.path = path
        self.reduction = reduction
        self.calc_properties(path)

    @property
    def reduction(self):
        """Get reduction mode"""
        return self._reduction

    @reduction.setter
    def reduction(self, mode: str):
        """Set mode for reduction of batch-wise loss to single value.

        Parameters
        ----------
        mode : str
            Reduction mode

        Raises
        ------
        TypeError
            Invalid loss given
        """
        if mode == "none":
            self._reduction = torch.nn.Identity()
        elif mode == "mean":
            self._reduction = torch.mean
        elif mode == "sum":
            self._reduction = torch.sum
        else:
            raise TypeError(f"Reduction mode '{mode}' unknown.")

    def forward(
        self, input: Tensor, target: Tensor, label: List[str], n_partner: Tensor
    ) -> Tensor:
        """Calculate WTMAD2 batch loss

        Parameters
        ----------
        input : Tensor
            Batch of input values
        target : Tensor
            Batch of target values
        label : List[str]
            List of subset labels
        n_partner : Tensor
            List indicating the number of partners per batched reaction

        Returns
        -------
        Tensor
            Reduced WTMAD2 loss
        """

        # get reaction-partner idx
        p_idx = torch.cumsum(n_partner, dim=0) - 1

        # contract labels
        label = [label[i] for i in p_idx]

        # create vector of subset values from label
        avgs = torch.tensor([self.subsets[l]["avg"].item() for l in label])

        # absolute error for each sample
        error = F.l1_loss(input, target, reduction="none")

        # wtmad2 scaling
        wtmad2 = torch.div(self.total_avg, avgs) * error
        return self.reduction(wtmad2)

    def calc_properties(self, path: Path):
        """Calculate GMTKN55 properties dynamically. Update instance properties.

        Parameters
        ----------
        path : str
            Relative path to directory containing GMTKN55 data
        """

        # load data
        dataset = get_gmtkn55_dataset(path, file_reactions= "reactions.json", file_samples= "samples.json")
        d = {}

        # collect values
        for r in dataset.reactions:
            subset = r.uid.split("_")[0]
            if subset not in d.keys():
                d[subset] = {
                    "ref": torch.tensor(0.0),
                    "count": torch.tensor(0.0),
                }
            d[subset]["ref"] += r.eref.abs()
            d[subset]["count"] += torch.tensor(1.0)

        # update self
        self.subsets = {
            subset: {
                "avg": d[subset]["ref"] / d[subset]["count"],
                "count": d[subset]["count"],
            }
            for subset in d
        }

        # average of all subset averages
        self.total_avg = torch.tensor(0.0)
        for _, subset in self.subsets.items():
            self.total_avg += subset["avg"]
        self.total_avg /= len(self.subsets)
