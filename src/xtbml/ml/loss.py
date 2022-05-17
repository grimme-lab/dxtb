""" Module for custom loss functions."""


import copy
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import List

from ..typing import Tensor
from ..utils import dict_reorder
from ..data.dataset import get_gmtkn_dataset


def get_gmtkn_ref_values(
    path: Path = Path(Path.cwd(), "../data/GMTKN55-main"), name: str = ".average"
) -> dict:
    """Get reference values for GMTKN-55 subset averages.

    Parameters
    ----------
    path : Path, optional
        Path to GMTKN-55 folder, by default Path(Path.cwd(), "../data/GMTKN55-main")
    name : str, optional
        Name of the file(s) containing the average energy per subset, by default ".average"

    Returns
    -------
    dict
        Dictionary conatining the average energy for each subset. Keys sorted alphabetically
    """
    d = {}
    for root, dirs, files in os.walk(path):
        if name in files:
            with open(os.path.join(root, name), "r") as file:
                d[os.path.basename(root)] = torch.tensor(float(file.read()))
    # no seperate folder
    d["BH76RC"] = {".numen": torch.tensor(30.0), ".average": torch.tensor(21.39)}.get(
        name
    )
    return dict_reorder(d)


gmtkn_ref = {
    "ACONF": {"count": 15.0, "avg": 1.83},
    "ADIM6": {"count": 6.0, "avg": 3.36},
    "AHB21": {"count": 21.0, "avg": 22.49},
    "AL2X6": {"count": 6.0, "avg": 35.88},
    "ALK8": {"count": 8.0, "avg": 62.6},
    "ALKBDE10": {"count": 10.0, "avg": 100.69},
    "Amino20x4": {"count": 80.0, "avg": 2.44},
    "BH76": {"count": 76.0, "avg": 18.61},  # "BH76": 19.40 / 106
    "BH76RC": {"count": 30.0, "avg": 21.39},
    "BHDIV10": {"count": 10.0, "avg": 45.33},
    "BHPERI": {"count": 26.0, "avg": 20.87},
    "BHROT27": {"count": 27.0, "avg": 6.27},
    "BSR36": {"count": 36.0, "avg": 16.2},
    "BUT14DIOL": {"count": 64.0, "avg": 2.8},
    "C60ISO": {"count": 9.0, "avg": 98.25},
    "CARBHB12": {"count": 12.0, "avg": 6.04},
    "CDIE20": {"count": 20.0, "avg": 4.06},
    "CHB6": {"count": 6.0, "avg": 26.79},
    "DARC": {"count": 14.0, "avg": 32.47},
    "DC13": {"count": 13.0, "avg": 54.98},
    "DIPCS10": {"count": 10.0, "avg": 654.26},
    "FH51": {"count": 51.0, "avg": 31.01},
    "G21EA": {"count": 25.0, "avg": 33.62},
    "G21IP": {"count": 36.0, "avg": 257.61},
    "G2RC": {"count": 25.0, "avg": 51.26},
    "HAL59": {"count": 59.0, "avg": 4.59},
    "HEAVY28": {"count": 28.0, "avg": 1.24},
    "HEAVYSB11": {"count": 11.0, "avg": 58.02},
    "ICONF": {"count": 17.0, "avg": 3.27},
    "IDISP": {"count": 6.0, "avg": 14.22},
    "IL16": {"count": 16.0, "avg": 109.05},
    "INV24": {"count": 24.0, "avg": 31.85},
    "ISO34": {"count": 34.0, "avg": 14.57},
    "ISOL24": {"count": 24.0, "avg": 21.92},
    "MB16-43": {"count": 43.0, "avg": 414.73},  # 468.39
    "MCONF": {"count": 51.0, "avg": 4.97},
    "NBPRC": {"count": 12.0, "avg": 27.71},
    "PA26": {"count": 26.0, "avg": 189.05},
    "PArel": {"count": 20.0, "avg": 4.63},
    "PCONF21": {"count": 18.0, "avg": 1.62},
    "PNICO23": {"count": 23.0, "avg": 4.27},
    "PX13": {"count": 13.0, "avg": 33.36},
    "RC21": {"count": 21.0, "avg": 35.7},
    "RG18": {"count": 18.0, "avg": 0.58},
    "RSE43": {"count": 43.0, "avg": 7.6},
    "S22": {"count": 22.0, "avg": 7.3},
    "S66": {"count": 66.0, "avg": 5.47},
    "SCONF": {"count": 17.0, "avg": 4.6},
    "SIE4x4": {"count": 16.0, "avg": 33.72},
    "TAUT15": {"count": 15.0, "avg": 3.05},
    "UPU23": {"count": 23.0, "avg": 5.72},
    "W4-11": {"count": 140.0, "avg": 306.91},
    "WATER27": {"count": 27.0, "avg": 81.14},  # 81.17
    "WCPT18": {"count": 18.0, "avg": 34.99},
    "YBDE18": {"count": 18.0, "avg": 49.28},
}


class WTMAD2Loss(torch.nn.Module):
    """Calculate the weighted total mean absolute deviation, as defined in

    - L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich,A. Najibi, Asim, S. Grimme,
      *Phys. Chem. Chem. Phys.*, **2017**, 19, 48, 32184-32215.
      (`DOI <http://dx.doi.org/10.1039/C7CP04913G>`__)
    """

    def __init__(
        self,
        rel_path: str = None,
        reduction: str = "mean",
    ) -> None:
        """_summary_

        Parameters
        ----------
        rel_path : str, optional
            Relative path of GMTKN-55 directory, by default None
        reduction : str, optional
            Reduction of batch-wise loss to single value, by default "mean"

        Raises
        ------
        TypeError
            Invalid loss given
        """
        super(WTMAD2Loss, self).__init__()

        if rel_path is None:
            # hardcoded values
            self.subsets = copy.deepcopy(gmtkn_ref)
            for k in self.subsets:
                # number of reactions in each subset
                self.subsets[k]["count"] = torch.tensor(self.subsets[k]["count"])
                # average energy for each subset
                self.subsets[k]["avg"] = torch.tensor(round(self.subsets[k]["avg"], 2))
            self.total_avg = torch.Tensor([56.84])
        else:
            # calculate properties dynamically
            self.rel_path = rel_path
            self.calc_properties(self.rel_path)

        if reduction == "none":
            self.reduction = torch.nn.Identity()
        elif reduction == "mean":
            self.reduction = torch.mean
        elif reduction == "sum":
            self.reduction = torch.sum
        else:
            raise TypeError

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
        counts = torch.tensor([self.subsets[l]["count"].item() for l in label])
        avgs = torch.tensor([self.subsets[l]["avg"].item() for l in label])

        # pytorchs' mad is not the MAD we usually use, our MAD is actually MAE
        mae = F.l1_loss(input, target, reduction="none")

        wtmad2 = torch.div(counts * self.total_avg, avgs) * mae / len(self.subsets)

        return self.reduction(wtmad2)

    def calc_properties(self, rel_path: str):
        """Calculate GMTKN-55 properties dynamically. Update instance properties.

        Parameters
        ----------
        rel_path : str
            Relative path to directory containing GMTKN-55 data
        """

        # load data
        dataset = get_gmtkn_dataset(rel_path)
        d = {}
        total_avg = torch.Tensor([0])

        # collect values
        for r in dataset.reactions:
            subset = r.uid.split("_")[0]
            if subset not in d.keys():
                d[subset] = {
                    "ref": torch.Tensor([0]),
                    "count": torch.Tensor([0]),
                }
            d[subset]["ref"] += r.eref.abs()
            d[subset]["count"] += 1
            total_avg += r.eref.abs()

        # update self
        self.subsets = {
            subset: {
                "count": d[subset]["count"],
                "avg": d[subset]["ref"] / d[subset]["count"],
            }
            for subset in d
        }
        self.total_avg = total_avg / len(dataset)
