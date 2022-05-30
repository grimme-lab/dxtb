from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor


from ..data.dataset import get_gmtkn_dataset


class TreeRegressor:
    """Class for training and predicting for regression task utilizing decision-tree like methods."""

    def __init__(
        self, features: pd.DataFrame, target: str, model: Union[RegressorMixin, str]
    ) -> None:

        self.features, self.labels, self.feature_list = TreeRegressor.prep_data(
            data=features, target=target
        )

        if isinstance(model, str):
            self.model = self.get_model(model)
        else:
            self.model = model

    @staticmethod
    def get_data() -> pd.DataFrame:
        """As default get the GMTKN55 dataset.

        Returns
        -------
        pd.DataFrame
            GMTKN55 dataset in pandas dataframe format
        """
        root = Path(__file__).resolve().parents[3]
        dataset = get_gmtkn_dataset(Path(root, "data"))
        return dataset.to_df()

    @staticmethod
    def prep_data(data: pd.DataFrame, target: str):

        # remove non-numerical columns
        features = data.drop("subset", axis=1)

        # one-hot encode categorical data
        features = pd.get_dummies(features)

        labels = np.array(features[target])
        features = features.drop(target, axis=1)
        feature_list = list(features.columns)
        features = np.array(features)

        return features, labels, feature_list

    @staticmethod
    def get_train_test(
        features: np.array, labels: np.array, test_size: float = 0.25
    ) -> np.array:
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        return train_features, test_features, train_labels, test_labels

    @staticmethod
    def get_model(key: str) -> RegressorMixin:
        d = {
            "bdt": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "rf": RandomForestRegressor(n_estimators=1000, random_state=42),
            "lgbm": LGBMRegressor(),
        }
        return d[key]

    def train(self) -> None:
        """Train the model invoking sklearn workflow."""
        self.model.fit(self.features, self.labels)

    def predict(self, test_features) -> np.ndarray:
        """Test the model invoking sklearn workflow."""
        predictions = self.model.predict(test_features)
        return predictions

    def feature_importance(self, verbose: bool = True) -> List[Tuple[str, float]]:
        """Get feature importance for the given model.

        Parameters
        ----------
        verbose : bool, optional
            Print feature importance to stdout, by default True.

        Returns
        -------
        List[Tuple[str, float]]
            Sorted feature importances by key and relative importance.
        """

        # Get numerical feature importances
        importances = list(self.model.feature_importances_)
        # List of tuples with variable and importance
        feature_importances = [
            (feature, round(importance, 2))
            for feature, importance in zip(self.feature_list, importances)
        ]
        # Sort the feature importances by most important first
        feature_importances = sorted(
            feature_importances, key=lambda x: x[1], reverse=True
        )
        if verbose:
            [
                print("Variable: {:20} Importance: {}".format(*pair))
                for pair in feature_importances
            ]
        return feature_importances
