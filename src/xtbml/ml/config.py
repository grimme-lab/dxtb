""" A collection of configuration classes """

from pydantic import BaseModel
from typing import Union


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
    """ Learning rate during training"""

    # Data setup configuration

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

    def get_model_cfg(self) -> dict:
        """Properties relevant for ML model architecture.

        Returns:
            dict: Dictionary containing configuration for ML model
        """
        return self.get_specific_properties("model_")

    def get_train_cfg(self) -> dict:
        """Properties relevant for training or PL trainer().

        Returns:
            dict: Dictionary containing configuration for training
        """
        return self.get_specific_properties("train_")

    def get_data_cfg(self) -> dict:
        """Properties relevant for data setup, incl. which data to choose and (pre-)transformation.

        Returns:
            dict: Dictionary containing configuration for data
        """
        return self.get_specific_properties("data_")
