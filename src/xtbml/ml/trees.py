"""
Decisiontree-based machine learning methods
============================================

Implementation of decisiontree algorthims for regression tasks.

Example
-------
>>> from xtbml.data.dataset import get_gmtkn55_dataset
>>> from xtbml.ml.trees import TreeRegressor
>>> from xtbml.ml.util import wtmad2
>>> # load data from disk
>>> features = TreeRegressor.get_data()
>>> # alternatively load from csv: features = pd.read_csv("df.csv")
>>> #
>>> tr = TreeRegressor(features=features, target="r_eref", model="bdt")
>>> tr.train()
>>> # for simplicity test on train data
>>> test_features, test_labels, _ = TreeRegressor.prep_data(features, target="r_eref")
>>> predictions = tr.predict(test_features)
>>> #
>>> df = features.copy()  # avoid SettingWithCopyWarning
>>> df = df[["subset", "r_egfn1", "r_eref"]]
>>> df["predictions"] = predictions.tolist()
>>> print(df)
   subset   r_egfn1  r_eref  predictions
0   ACONF  0.328462   0.598     0.598122
1   ACONF  0.297663   0.614     0.613967
2   ACONF  0.743076   0.961     0.961005
3   ACONF  1.761008   2.813     2.812976
4   ACONF  0.276938   0.595     0.595119
5   ACONF  0.277891   0.604     0.604012
6   ACONF  0.682718   0.934     0.934077
7   ACONF  0.554463   1.178     1.177967
8   ACONF  0.617511   1.302     1.301865
9   ACONF  1.018668   1.250     1.250103
10  ACONF  1.693739   2.632     2.632014
11  ACONF  1.718680   2.740     2.739992
12  ACONF  1.932725   3.283     3.282962
13  ACONF  2.114138   3.083     3.082936
14  ACONF  3.569065   4.925     4.924882
>>> loss_gfn1 = wtmad2(df=df, colname_target="r_egfn1", colname_ref="r_eref")
>>> print("GFN-1: ", loss_gfn1)
GFN-1:  0.2078990332053096
>>> loss_tree = wtmad2(df=df, colname_target="predictions", colname_ref="r_eref")
>>> print("Tree: ", loss_tree)
Tree:  1.902131842981567e-05
"""

from typing import List, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor


from ..data.dataset import get_gmtkn55_dataset


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
        dataset = get_gmtkn55_dataset(Path(root, "data"))
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
