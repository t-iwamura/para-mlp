from typing import List

from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from sklearn.linear_model import Ridge

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant
from para_mlp.utils import rmse


class RILRM:
    """
    Rotation Invariant type Linear Regression Model
    """

    def __init__(
        self, model_params: ModelParams, kfold_structures: List[Structure]
    ) -> None:
        self._ri = RotationInvariant(model_params)
        self._ridge = Ridge(model_params.alpha)

        # Make feature
        self._x = self._make_feature(kfold_structures)

    def _make_feature(self, structure_set: List[Structure]) -> NDArray:
        return self._ri(structure_set)

    def __call__(self, structure_set: List[Structure]) -> NDArray:
        self._x = self._make_feature(structure_set)

        return self._ridge.predict(self._x)

    def train_and_validate(self, train_index, val_index, y_kfold):
        self._ridge.fit(self._x[train_index], y_kfold[train_index])

        y_predict = self._ridge.predict(self._x[val_index])
        y_target = y_kfold[val_index]

        return rmse(y_predict, y_target)
