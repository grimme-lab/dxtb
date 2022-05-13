""" Module for custom loss functions."""


import copy
from pathlib import Path
import os
import torch
from torch.nn.modules.loss import _Loss, _WeightedLoss

import torch.nn.functional as F

import pandas as pd
from typing import Optional, List

from ..typing import Tensor
from ..utils import dict_reorder
from .training import get_gmtkn_dataset


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
        super(WTMAD2Loss, self).__init__()

        # relative path of GMTKN-55 directory

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
            raise NotImplementedError

    def forward(
        self, input: Tensor, target: Tensor, label: List[str], n_partner: Tensor
    ) -> Tensor:

        # list of subset label
        # list indicating the number of partners per batched reaction

        verbose = False

        # legacy support -- broadcasting increases required RAM
        broadcast_to_reactants = False

        if verbose:
            print("inside")
            print(label)
            print(n_partner)

        # get reaction-partner idx
        p_idx = torch.cumsum(n_partner, dim=0) - 1

        if not broadcast_to_reactants:
            # contract labels
            label = [label[i] for i in p_idx]

        #####################
        # create vector of subset values from label
        counts = torch.tensor([self.subsets[l]["count"].item() for l in label])
        avgs = torch.tensor([self.subsets[l]["avg"].item() for l in label])

        if verbose:
            print("preparing weights")
            print("counts: ", counts)
            print("avgs: ", avgs)

            print("input: ", input)
            print("target: ", target)

        #####################

        """# pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        mue = (ref - target).abs().mean()
        wtmad += len_sub * AVG / avg_subset * mue"""

        # pytorchs' mad is not the MAD we usually use, our MAD is actually MAE
        mae = F.l1_loss(input, target, reduction="none")

        if broadcast_to_reactants:
            # broadcast to reaction partners
            # NOTE: works with different number of partners for reactions in batch
            mae = mae.repeat_interleave(n_partner)

        if verbose:
            print("mae", mae, mae.shape)

        # abc = (
        #     self.subsets[key]["count"] * self.total_avg / self.subsets[key]["avg"] * mae
        # ) / len(self.subsets)
        # print(abc)

        if verbose:
            print(len(self.subsets), self.total_avg, mae)
            print(counts * self.total_avg)
            print((counts * self.total_avg).shape)
            print(torch.div((counts * self.total_avg), avgs))
            print(torch.div((counts * self.total_avg), avgs).shape)

            print(torch.div(counts * self.total_avg, avgs) * mae)
            print((torch.div(counts * self.total_avg, avgs) * mae).shape)

        wtmad2 = torch.div(counts * self.total_avg, avgs) * mae / len(self.subsets)
        if verbose:
            print("wtmad2")
            print(wtmad2)
            print(wtmad2.shape)

        if broadcast_to_reactants:
            # contract to reactions
            wtmad2 = wtmad2[p_idx]

        if verbose:
            print("wtmad2 2")
            print(wtmad2)
            print(wtmad2.shape)
            print("input", input.size()[0])
        assert input.size()[0] == len(
            wtmad2
        ), "Consecutive samples belong to different reactions."

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


def wtmad2(
    df: pd.DataFrame,
    colname_target: str,
    colname_ref: str,
    set_column: str = "subset",
    verbose: bool = True,
) -> float:

    AVG = 56.84

    subsets = df.groupby([set_column])
    subset_names = df[set_column].unique()

    wtmad = 0
    for name in subset_names:
        sdf = subsets.get_group(name)
        ref = sdf[colname_ref]
        target = sdf[colname_target]

        # number of reactions in each subset
        count = target.count()

        # compute average reaction energy for each subset
        avg_subset = ref.mean()

        # pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        # https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/generic.py#L10813
        mue = (ref - target).abs().mean()
        wtmad += count * AVG / avg_subset * mue

        if verbose:
            print(f"Subset {name} ({count} entries): MUE {mue:.3f}")

    return wtmad / len(df.index)


class NLLLoss(_WeightedLoss):
    r"""The negative log likelihood loss. It is useful to train a classification"""
    __constants__ = ["ignore_index", "reduction"]
    ignore_index: int

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
