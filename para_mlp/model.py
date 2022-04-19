from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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
        self._x = self._make_feature(kfold_structures, make_scaler=True)

    def _make_feature(
        self, structure_set: List[Structure], make_scaler: bool = False
    ) -> NDArray:
        """Make the feature matrix from given structure set

        Args:
            structure_set (List[Structure]): set of structures used
            make_scaler (bool): Whether to make scaler. Defaults to False.

        Returns:
            NDArray: feature matrix
        """
        x = self._ri(structure_set)

        if self._use_spin:
            spin_feature_matrix = self._sf(structure_set)
            x = np.hstack((x, spin_feature_matrix))

        if make_scaler:
            eid_end = len(structure_set)
            self._scaler = StandardScaler(with_mean=False).fit(x[:eid_end])

        x = self._scaler.transform(x, copy=False)

        return x

    @property
    def x(self) -> NDArray:
        return self._x

    def train(self, train_index: List[int], y_kfold: NDArray) -> None:
        self._ridge.fit(self._x[train_index], y_kfold[train_index])

    def train_and_validate(
        self, train_index: List[int], valid_index: List[int], y_kfold: NDArray
    ) -> float:
        """Do training and validation

        Args:
            train_index (List[int]): index list of training matrix
            valid_index (List[int]): index list of validation matrix
            y_kfold (NDArray): objective variable

        Returns:
            float: validation score(RMSE)
        """
        self._ridge.fit(self._x[train_index], y_kfold[train_index])

        y_predict = self._ridge.predict(self._x[valid_index])
        y_target = y_kfold[valid_index]

        return rmse(y_predict, y_target)

    def predict(self, structure_set: List[Structure] = None) -> NDArray:
        """Predict total energy of given structures and forces on atoms in given structures

        Args:
            structure_set (List[Structure]): structure set

        Returns:
            NDArray: objective variable
        """
        if structure_set is not None:
            self._x = self._make_feature(structure_set)

        return self._ridge.predict(self._x)
