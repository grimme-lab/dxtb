import torch
from pathlib import Path
import pandas as pd

from ..data.dataset import get_gmtkn_dataset
from .util import load_model_from_cfg

""" Load ML model from disk and conduct basic evaluation. """


def evaluate():
    """Simple evaluation of performance of a given model."""

    root = Path(__file__).resolve().parents[3]
    torch.set_printoptions(precision=2, sci_mode=False)

    # load data
    dataset = get_gmtkn_dataset(Path(root, "data"))

    # bookkeeping
    bset = [r.uid.split("_")[0] for r in dataset.reactions]
    # cc = [r.uid for r in dataset.reactions]
    # print(f"Dataset contains {len(cc)} reactions: {cc}")

    # remove all samples with missing values
    idxs = [i for i in range(len(dataset)) if len(dataset[i][0]) == 0]
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 1})

    # load model
    model_id = "202205242315"
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_optimizer": "Adam",
        "training_loss_fn": "L1Loss",
        "training_loss_fn_path": Path(root, "data"),
        "training_lr": 0.01,
        "epochs": 3,
        "model_state_dict": torch.load(f"{root}/models/{model_id}_model.pt"),
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

        print(
            f"Enn: {enn[i]:.2f} | Eref: {eref[i]:.2f} | Egfn1: {egfn1[i]:.2f} | Samples: {batched_reaction.partners}"
        )

    df = pd.DataFrame(
        list(zip(bset, eref, egfn1, enn)), columns=["subset", "Eref", "Egfn1", "Enn"]
    )
    df["dEgfn1"] = (df["Eref"] - df["Egfn1"]).abs()
    df["dEnn"] = (df["Eref"] - df["Enn"]).abs()
    print(df)
    print(df[["dEgfn1", "dEnn"]].describe())
    print("")
    df.to_csv(f"{model_id}_eval.csv")

    return bset, eref, egfn1, enn
