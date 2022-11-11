######################################################################
####### Script containing utility functions for model handling #######
####### such as init, saving and loading of pytorch models.    #######
######################################################################

from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from dxtb.ml.model_gnn import GNN

from .model import Simple_Net
from .loss import WTMAD2Loss


def get_architecture(name: str):
    """Returns architecture for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the architecture

    Returns:
        [nn.Module]: Architecture of the model
    """

    architecture_dict = {
        "Simple_Net": Simple_Net,
        "GNN": GNN,
    }

    return architecture_dict.get(name, Simple_Net)


def get_optimizer(name: str) -> torch.optim.Adam | torch.optim.SGD:
    """Returns optimiser for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the optimiser

    Returns:
        [nn.optim]: optimiser of the model
    """

    optimiser_dict = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
    }

    return optimiser_dict.get(name, torch.optim.Adam)


def get_scheduler(
    name: str, optimizer
) -> optim.lr_scheduler.StepLR | optim.lr_scheduler.ReduceLROnPlateau | optim.lr_scheduler.CyclicLR:
    """Returns learning rate scheduler for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the scheduler

    Returns:
        [nn.optim.lr_scheduler]: scheduler of the model
    """

    const = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

    # Further options see: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler_dict = {
        "Const": const,
        "StepLR": optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5),
        "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, verbose=True
        ),
        "CyclicLR": optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=0.001,
            max_lr=0.1,
            step_size_up=2000,
            step_size_down=None,
            mode="triangular",
            gamma=1.0,
            cycle_momentum=False,
        ),
    }

    return scheduler_dict.get(name, const)


def get_loss_fn(
    name: str, path: Optional[Path] = None
) -> nn.L1Loss | nn.MSELoss | WTMAD2Loss:
    """Returns loss function for the given choice from the predefined archetypes.

    Parameters
    ----------
    name : str
        Name-tag for the loss function
    path : Optional[Path], optional
        Absolute path of GMTKN55 directory required for WTMAD2Loss, by default None

    Returns
    -------
    nn.L1Loss | nn.MSELoss | WTMAD2Loss
        Loss function of model
    """

    loss_fn_dict = {
        "L1Loss": nn.L1Loss(reduction="mean"),
        "L2Loss": nn.MSELoss(reduction="mean"),
        "WTMAD2Loss": WTMAD2Loss(path, reduction="mean"),
    }

    return loss_fn_dict.get(name, nn.L1Loss(reduction="mean"))


def load_model_from_cfg(cfg: dict, load_state=True):
    """Loads the model, optimiser and loss function
    from a given config file. If load_state set
    to False, no checkpoint is loaded.
    """

    architecture = get_architecture(cfg.get("model_architecture"))
    optimizer = get_optimizer(cfg.get("training_optimizer"))
    loss_fn = get_loss_fn(cfg.get("training_loss_fn"), cfg.get("training_loss_fn_path"))

    # load model parameters
    model = architecture(cfg)
    if load_state:
        model.load_state_dict(cfg["model_state_dict"])

    optimizer = optimizer(
        model.parameters(),
        lr=cfg.get("training_lr"),
        weight_decay=cfg.get("training_weight_decay", 0),
    )
    if load_state and "optimizer_state_dict" in cfg:
        optimizer.load_state_dict(cfg["optimizer_state_dict"])

    scheduler = get_scheduler(cfg.get("training_scheduler", None), optimizer)

    return model, optimizer, loss_fn, scheduler


def wtmad2(
    df: pd.DataFrame,
    colname_target: str,
    colname_ref: str,
    set_column: str = "subset",
    verbose: bool = True,
    calc_subsets=False,
) -> List[float]:
    """Calculate the weighted total mean absolute deviation, as defined in

    - L. Goerigk, A. Hansen, C. Bauer, S. Ehrlich,A. Najibi, Asim, S. Grimme,
      *Phys. Chem. Chem. Phys.*, **2017**, 19, 48, 32184-32215.
      (`DOI <http://dx.doi.org/10.1039/C7CP04913G>`__)

    Args:
        df (pd.DataFrame): Dataframe containing target and reference energy values.
        colname_target (str): Name of target column.
        colname_ref (str): Name of reference column.
        set_column (str, optional): Name of column defining the subset association. Defaults to "subset".
        verbose (bool, optional): Allows for printout of subset-wise MAD. Defaults to "False".

    Returns:
        List[float]: Weighted total mean absolute error of subsets and whole benchmark.
    """

    AVG = 57.82

    basic = [
        "W4-11",
        "G21EA",
        "G21IP",
        "DIPCS10",
        "PA26",
        "SIE4x4",
        "ALKBDE10",
        "YBDE18",
        "AL2X6",
        "HEAVYSB11",
        "NBPRC",
        "ALK8",
        "RC21",
        "G2RC",
        "BH76RC",
        "FH51",
        "TAUT15",
        "DC13",
    ]

    reactions = [
        "MB16-43",
        "DARC",
        "RSE43",
        "BSR36",
        "CDIE20",
        "ISO34",
        "ISOL24",
        "C60ISO",
        "PArel",
    ]

    barriers = ["BH76", "BHPERI", "BHDIV10", "INV24", "BHROT27", "PX13", "WCPT18"]

    intra = [
        "IDISP",
        "ICONF",
        "ACONF",
        "Amino20x4",
        "PCONF21",
        "MCONF",
        "SCONF",
        "UPU23",
        "BUT14DIOL",
    ]

    inter = [
        "RG18",
        "ADIM6",
        "S22",
        "S66",
        "HEAVY28",
        "WATER27",
        "CARBHB12",
        "PNICO23",
        "HAL59",
        "AHB21",
        "CHB6",
        "IL16",
    ]

    basic_wtmad, reactions_wtmad, barriers_wtmad, intra_wtmad, inter_wtmad = (
        0,
        0,
        0,
        0,
        0,
    )
    basic_count, reactions_count, barriers_count, intra_count, inter_count = (
        0,
        0,
        0,
        0,
        0,
    )

    subsets = df.groupby([set_column])
    subset_names = df[set_column].unique()

    wtmad = 0
    for name in subset_names:

        sdf = subsets.get_group(name)
        ref = sdf[colname_ref]
        target = sdf[colname_target]

        # number of reactions in each subset
        count = target.count()
        # print(name, count)
        # print(target)

        # compute average reaction energy for each subset
        avg_subset = ref.abs().mean()

        # pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        # https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/generic.py#L10813
        mae = (ref - target).abs().mean()

        if calc_subsets is True:
            if name in basic:
                basic_wtmad += count * AVG / avg_subset * mae
                basic_count += count
            elif name in reactions:
                reactions_wtmad += count * AVG / avg_subset * mae
                reactions_count += count
            elif name in barriers:
                barriers_wtmad += count * AVG / avg_subset * mae
                barriers_count += count
            elif name in intra:
                intra_wtmad += count * AVG / avg_subset * mae
                intra_count += count
            elif name in inter:
                inter_wtmad += count * AVG / avg_subset * mae
                inter_count += count
            else:
                raise ValueError(f"Subset '{name}' not found in lists.")

        wtmad += count * AVG / avg_subset * mae

        if verbose:
            print(
                f"Subset {name} ({count} entries): MUE {mae:.3f} | count {count} | avg: {avg_subset} | AVG: {AVG}"
            )

    if calc_subsets is True:
        return [
            basic_wtmad / basic_count,
            reactions_wtmad / reactions_count,
            barriers_wtmad / barriers_count,
            intra_wtmad / intra_count,
            inter_wtmad / inter_count,
            wtmad / len(df.index),
        ]
    else:
        return wtmad / 1505
