from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from sklearn.linear_model import Ridge

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant, SpinFeaturizer
from para_mlp.utils import rmse


class RILRM:
    """
    Rotation Invariant type Linear Regression Model
    """

    def __init__(
        self, model_params: ModelParams, kfold_structures: List[Structure]
    ) -> None:
        self._use_spin = model_params.use_spin

        self._ri = RotationInvariant(model_params)
        if self._use_spin:
            self._sf = SpinFeaturizer(model_params)

        self._ridge = Ridge(model_params.alpha)

        # Make feature
        self._x = self._make_feature(kfold_structures)

    def _make_feature(self, structure_set: List[Structure]) -> NDArray:
        """Make the feature matrix from given structure set

        Args:
            structure_set (List[Structure]): set of structures used

        Returns:
            NDArray: feature matrix
        """
        x = self._ri(structure_set)

        if self._use_spin:
            spin_feature_matrix = self._sf(structure_set)
            x = np.hstack((x, spin_feature_matrix))

        return x

    def train_and_validate(
        self, train_index: List[int], val_index: List[int], y_kfold: NDArray
    ) -> float:
        """Do training and validation

        Args:
            train_index (List[int]): index list of training matrix
            val_index (List[int]): index list of validation matrix
            y_kfold (NDArray): objective variable

        Returns:
            float: validation score(RMSE)
        """
        self._ridge.fit(self._x[train_index], y_kfold[train_index])

        y_predict = self._ridge.predict(self._x[val_index])
        y_target = y_kfold[val_index]

        return rmse(y_predict, y_target)

    def predict(self, structure_set: List[Structure]) -> NDArray:
        """Predict total energy of given structures and forces on atoms in given structures

        Args:
            structure_set (List[Structure]): structure set

        Returns:
            NDArray: objective variable
        """
        self._x = self._make_feature(structure_set)

        return self._ridge.predict(self._x)
