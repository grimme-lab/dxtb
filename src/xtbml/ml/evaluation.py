import torch
import pandas as pd

from .training import get_gmtkn_dataset
from .util import load_model_from_cfg

""" Load ML model from disk and conduct basic evaluation. """


def evaluate():
    """Simple evaluation of performance of a given model."""

    torch.set_printoptions(precision=2, sci_mode=False)

    # load data
    dataset = get_gmtkn_dataset()

    # prune to constant number of partners
    np = [len(r.partners) for r in dataset.reactions]
    idxs = [i for i, x in enumerate(np) if x != 2]
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)
    cc = [r.uid for r in dataset.reactions]
    bset = [r.uid.split("_")[0] for r in dataset.reactions]
    print(f"Dataset contains {len(cc)} reactions: {cc}")

    # remove all samples with missing values
    idxs = [i for i in range(len(dataset)) if len(dataset[i][0]) == 0]
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 1})

    # load model
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_optimizer": "Adam",
        "training_loss_fn": "L1Loss",
        "training_lr": 0.01,
        "epochs": 3,
        "model_state_dict": torch.load("../models/202205041550_model.pt"),
    }
    model, _, loss_fn, _ = load_model_from_cfg(cfg_ml)
    model.eval()

    # evaluate model
    eref, egfn1, enn = [], [], []
    for i, (batched_samples, batched_reaction) in enumerate(dl):

        y = model(batched_samples, batched_reaction)
        y_true = batched_reaction.eref

        eref.append(y_true.item())
        egfn1.append(batched_reaction.egfn1.item())
        enn.append(y.item())
        print(f"Enn: {enn[i]:.2f} | Eref: {eref[i]:.2f} | Egfn1: {egfn1[i]:.2f}")

    df = pd.DataFrame(
        list(zip(bset, eref, egfn1, enn)), columns=["subset", "Eref", "Egfn1", "Enn"]
    )
    df["dEgfn1"] = (df["Eref"] - df["Egfn1"]).abs()
    df["dEnn"] = (df["Eref"] - df["Enn"]).abs()
    print(df)

    print(df[["dEgfn1", "dEnn"]].describe())
    print("")
    print("WTMAD-2: Egfn1", wtmad2(df, "Egfn1", "Eref"))
    print("")
    print("WTMAD-2: Enn", wtmad2(df, "Enn", "Eref"))


def wtmad2(df, colname_target, colname_ref, set_column="subset"):
    AVG = 56.84
    subsets = df.groupby([set_column])
    subset_names = df[set_column].unique()

    # number of reactions in each subset
    counts = subsets.count()[colname_target]

    # compute average reaction energy for each subset
    avg_subsets = subsets.mean()[colname_ref]

    wtmad = 0
    for i in range(subsets.ngroups):
        sdf = subsets.get_group(subset_names[i])

        # pandas' mad is not the MAD we usually use, our MAD is actually MUE/MAE
        # https://github.com/pandas-dev/pandas/blob/v1.4.2/pandas/core/generic.py#L10813
        mue = (sdf[colname_ref] - sdf[colname_target]).abs().mean()
        wtmad += counts[i] * AVG / avg_subsets[i] * mue
        print(f"Subset {subset_names[i]} ({counts[i]} entries): MUE {mue:.3f}")

    return wtmad / len(df.index)
