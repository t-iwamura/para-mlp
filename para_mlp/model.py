import pickle
from pathlib import Path
from typing import Any, List, Tuple

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


def dump_model(model: Any, model_params: ModelParams, model_dir: str) -> None:
    """Dump model object and ModelParams dataclass

    Args:
        model (Any): model object
        model_params (ModelParams): ModelParams dataclass. Store model's parameter.
        model_dir (str): path to directory where given model is dumped
    """
    model_filepath = Path(model_dir) / "model.pkl"
    with model_filepath.open("wb") as f:
        pickle.dump((model, model_params), f)


def dump_model_as_lammps(model, model_dir: str) -> None:
    """Dump model as lammps file format

    Args:
        model (ModelParams): Trained ModelParams object
        model_dir (str, optional): The directory where potential file for lammps is
            saved. Defaults to "mlp.lammps".
    """
    content = make_content_of_lammps_file(model)

    lammps_file_path = Path(model_dir) / "mlp.lammps"
    with lammps_file_path.open("w") as f:
        f.write(content)


def make_content_of_lammps_file(model: RILRM) -> str:
    """Make content of potential file for lammps

    Args:
        model (ModelParams): Trained ModelParams object

    Returns:
        str: The content of potential file
    """
    model_params = model._ri._model_params
    lines = []

    lines.append("Fe # element\n")
    lines.append(f"{model_params.cutoff_radius} # cutoff\n")
    lines.append(f"{model_params.radial_func} # pair_type\n")
    lines.append(f"{model_params.feature_type} # des_type\n")
    lines.append(f"{model_params.polynomial_model} # model_type\n")
    lines.append(f"{model_params.polynomial_max_order} # max_p\n")
    lines.append(f"{model_params.lmax} # max_l\n")
    lines.append(f"{model_params.gtinv_order} # gtinv_order\n")
    for item in model_params.gtinv_lmax:
        lines.append(f"{item} ")
    lines.append(" # gtinv_maxl\n")
    for item in model_params.gtinv_sym:
        lines.append(f"{int(item)} ")
    lines.append(" # gtinv_sym\n")
    lines.append(f"{model._ridge.coef_.shape[0]} # number of regression coefficients\n")
    for item in model._ridge.coef_:
        lines.append(f"{item:15.15e} ")
    lines.append(" # reg. coeffs.\n")
    for item in model._scaler.scale_:
        lines.append(f"{item:15.15e} ")
    lines.append(" # scales\n")
    lines.append(f"{len(model_params.radial_params)} # number of parameters\n")
    for item in model_params.radial_params:  # type: ignore
        lines.append(  # type: ignore
            f"{item[0]:15.15f} {item[1]:15.15f} # pair func. params\n"  # type: ignore
        )  # type: ignore
    lines.append("5.585000000000000e+01  # atomic mass\n")
    lines.append("False # electrostatic\n")

    content = "".join(lines)

    return content


def load_model(model_dir: str) -> Tuple[Any, ModelParams]:
    """Load model object and ModelParams dataclass

    Args:
        model_dir (str): path to directory where the model is dumped

    Returns:
        Tuple[Any, ModelParams]: model object and ModelParams dataclass
    """
    model_filepath = Path(model_dir) / "model.pkl"
    with model_filepath.open("rb") as f:
        model, model_params = pickle.load(f)

    return model, model_params
