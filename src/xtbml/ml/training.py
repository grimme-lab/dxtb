""" Simple training pipline for pytoch ML model. """

import datetime
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd

from xtbml.ml.util import load_model_from_cfg
from xtbml.data.dataset import ReactionDataset


def train():
    """Trains the model."""

    # load GMTKN55 from disk
    dataset = get_gmtkn_dataset()

    # TODO: extend padding to number of parameters
    # prune to constant number of partners
    if False:
        np = [len(r.partners) for r in dataset.reactions]
        idxs = [i for i, x in enumerate(np) if x != 2]
        for i in reversed(idxs):
            dataset.rm_reaction(idx=i)
        cc = [r.uid for r in dataset.reactions]
        print(f"Dataset contains {len(cc)} reactions: {cc}")

    # TODO: apparently iterator not correctly implemented
    # for i, s in enumerate(dataset):
    #    print(dataset[i])

    # remove all samples with missing values
    idxs = [i for i in range(len(dataset)) if len(dataset[i][0]) == 0]
    for i in reversed(idxs):
        dataset.rm_reaction(idx=i)

    # setup dataloader
    dl = dataset.get_dataloader({"batch_size": 50})

    # set config for ML
    cfg_ml = {
        "model_architecture": "Basic_CNN",
        "training_optimizer": "Adam",
        "training_loss_fn": "L1Loss",
        "training_lr": 0.01,
        "epochs": 60,
    }

    # load components
    model, optimizer, loss_fn, scheduler = load_model_from_cfg(cfg_ml, load_state=False)

    # run training
    model.train()
    losses = []
    mads, mads_ref = [], []
    for i in range(cfg_ml["epochs"]):
        print(f"EPOCH {i}")
        losses_epoch = []
        mads_epoch, mads_ref_epoch = [], []
        for batched_samples, batched_reaction in dl:

            # some samples required in reaction are not available
            if batched_samples == []:
                continue

            # prediction based on QM features
            y = model(batched_samples, batched_reaction)

            # TODO: where to use reaction.egfn1?

            # reshape to avoid broadcasting
            # y = y.view(y.size(0), -1)
            y_true = batched_reaction.eref

            # optimize model parameter and feature parameter
            loss = loss_fn(y, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: optimise parameters back to initial positions
            #       (propagate gradients and update parameters)

            # print(f"loss: {loss} | y: {y} | y_true: {y_true}")

            # bookkeeping
            mad_fn = nn.L1Loss(reduction="mean")
            losses_epoch.append(loss.item())
            mads_epoch.append(mad_fn(y, y_true).item())
            mads_ref_epoch.append(mad_fn(batched_reaction.egfn1, y_true).item())

        # Loss per epoch is average of batch losses
        losses.append(sum(losses_epoch) / len(losses_epoch))
        mads.append(sum(mads_epoch) / len(mads_epoch))
        mads_ref.append(sum(mads_ref_epoch) / len(mads_ref_epoch))

    print("Finished training")

    df = pd.DataFrame(
        list(zip(losses, mads, mads_ref)), columns=["Losses", "MAD", "MAD_ref"]
    )

    # save model
    uid = f'{datetime.datetime.now().strftime("%Y%m%d%H%M")}'
    save_name = f"../models/{uid}"
    df.to_csv(save_name + "_df.csv", index=True)
    torch.save(model.state_dict(), save_name + "_model.pt")


def get_dataset(path_reactions: str, path_samples: str) -> ReactionDataset:
    """Return a preliminary dataset for setting up the workflow."""

    # load json from disk
    dataset = ReactionDataset.create_from_disk(
        path_reactions=path_reactions, path_samples=path_samples
    )

    # apply simple padding
    dataset.pad()

    return dataset


def get_gmtkn_dataset() -> ReactionDataset:
    """Return total gmtkn55 dataset."""
    rel = "../data"

    # # !use this once all features are available in one file
    # dataset = get_dataset(
    #     path_reactions=Path(Path.cwd(), rel, "reactions.json"),
    #     path_samples=Path(Path.cwd(), rel, "samples.json"),
    # )
    # assert len(dataset) == 1505
    # return dataset

    # load data
    d1 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-1.json"),
    )
    d2 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-2.json"),
    )
    d3 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-3.json"),
    )
    d4 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-4.json"),
    )
    d5 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-5.json"),
    )
    d6 = get_dataset(
        path_reactions=Path(Path.cwd(), rel, "reactions.json"),
        path_samples=Path(Path.cwd(), rel, "features-6.json"),
    )

    # merge
    dataset = ReactionDataset.merge(d1, d2)
    dataset = ReactionDataset.merge(dataset, d3)
    dataset = ReactionDataset.merge(dataset, d4)
    dataset = ReactionDataset.merge(dataset, d5)
    dataset = ReactionDataset.merge(dataset, d6)
    dataset.pad()

    assert len(dataset) == 6 * 1499

    # TODO: alternatively save to json

    return dataset
