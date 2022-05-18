######################################################################
####### Script containing utility functions for model handling #######
####### such as init, saving and loading of pytorch models.    #######
######################################################################

import torch
import torch.nn as nn
import torch.optim as optim

from .model import Basic_CNN
from .loss import WTMAD2Loss


def get_architecture(name):
    """Returns architecture for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the architecture

    Returns:
        [nn.Module]: Architecture of the model
    """

    architecture_dict = {
        "Basic_CNN": Basic_CNN,
    }

    return architecture_dict.get(name)


def get_optimizer(name):
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

    return optimiser_dict.get(name)


def get_scheduler(name, optimizer):
    """Returns learning rate scheduler for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the scheduler

    Returns:
        [nn.optim.lr_scheduler]: scheduler of the model
    """

    # Further options see: https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler_dict = {
        "Const": optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0),
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

    return scheduler_dict.get(name)


def get_loss_fn(name):
    """Returns loss function for the given choice
        from the predefined archetypes.

    Args:
        name (str): Name-tag for the loss function

    Returns:
        [nn.Loss]: loss function of the model
    """

    loss_fn_dict = {
        "L1Loss": nn.L1Loss(reduction="mean"),
        "L2Loss": nn.MSELoss(reduction="mean"),
        "WTMAD2Loss": WTMAD2Loss(reduction="mean"),
    }

    return loss_fn_dict.get(name)


def load_model_from_cfg(cfg: dict, load_state=True):
    """Loads the model, optimiser and loss function
    from a given config file. If load_state set
    to False, no checkpoint is loaded.
    """

    architecture = get_architecture(cfg.get("model_architecture"))
    optimizer = get_optimizer(cfg.get("training_optimizer"))
    loss_fn = get_loss_fn(cfg.get("training_loss_fn"))

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
