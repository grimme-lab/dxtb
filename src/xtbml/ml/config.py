""" A collection of configuration classes """

from typing import List, Union

import pytorch_lightning as pl
from pydantic import BaseModel
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.base import Callback

# TODO:
# * for cli parsing, see: https://github.com/mpkocher/pydantic-cli

# NOTE: for loading from disk (eg. JSON) use: Lightning_Configuration.parse_file(path)


class Lightning_Configuration(BaseModel):
    """Configuration utilised to set up pytorch lightning modules.
    This includes preparing a Trainer(), model and the data."""

    uid: Union[int, str] = "default"
    """ Unique identifier for configuration """

    # ML model configuration
    model_hidden: int = 10
    """ Size of hidden layer """

    # PL trainer configuration
    train_lr: float = 0.01
    """ Learning rate during training """
    train_max_epochs: int = 3
    """ Number of epochs """
    train_callbacks: list[Callback] = [LearningRateMonitor(logging_interval="epoch")]
    """ Callbacks for training """
    train_log_every_n_steps: int = 1
    """ Update logger every n-steps """
    train_no_wandb: bool = True
    """ Toggling weights and biases """

    # Data setup configuration
    # ... tbd ...

    class Config:
        # allow other properties
        extra = "allow"
        # allow for object-typed properties
        arbitrary_types_allowed = True

    def __init__(self, **data) -> None:
        super().__init__(**data)
        self.setup_train_default()

    def get_specific_properties(self, key: str) -> dict:
        """Returns all entries that start with value of key (and 'uid').

        Args:
            key (str): Key to identify properties

        Returns:
            dict: Properties starting with key or being 'uid'
        """
        specific_cfg = self.dict()
        for k in list(specific_cfg.keys()):
            if not k.startswith(key) and k != "uid":
                del specific_cfg[k]
        return specific_cfg

    def remove_key(self, d: dict, key: str) -> dict:
        """Remove leading key from given dictionary.

        Args:
            d (dict): Dictionary containing entries
            key (str): Key to be removed from beginning of dictionary keys

        Returns:
            dict: Renamned dictionary
        """
        return {k[len(key) :] if k.startswith(key) else k: v for k, v in d.items()}
        # return {k.removeprefix(key): v for k, v in d.items()} # python 3.9

    def get_model_cfg(self) -> dict:
        """Properties relevant for ML model architecture.

        Returns:
            dict: Dictionary containing configuration for ML model
        """
        return self.remove_key(self.get_specific_properties("model_"), "model_")

    def get_train_cfg(self) -> dict:
        """Properties relevant for training or PL trainer().

        Returns:
            dict: Dictionary containing configuration for training
        """
        return self.remove_key(self.get_specific_properties("train_"), "train_")

    def get_data_cfg(self) -> dict:
        """Properties relevant for data setup, incl. which data to choose and (pre-)transformation.

        Returns:
            dict: Dictionary containing configuration for data
        """
        return self.remove_key(self.get_specific_properties("data_"), "data_")

    def setup_train_default(self):
        """Set default pl trainer attributes. Does not update previously set values."""
        defaults = pl.Trainer.default_attributes()
        # NOTE: callbacks not included

        for k, v in defaults.items():
            key = f"train_{k}"
            # avoid overwriting
            if not hasattr(self, key):
                setattr(self, key, v)

    def parse_cli(self):

        raise NotImplementedError

        # parse arguments
        parser = ArgumentParser()
        parser.add_argument(
            "--no_wandb", dest="no_wandb", default=False, action="store_true"
        )
        parser = pl.Trainer.add_argparse_args(parser)
        args = parser.parse_args()
        print("args: ", args)
        trainer = pl.Trainer.from_argparse_args(args)
