import json
import pickle
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant, SpinFeaturizer


class RILRM:
    """
    Rotation Invariant type Linear Regression Model
    """

    def __init__(self, model_params: ModelParams) -> None:
        self._use_spin = model_params.use_spin

        self._ri = RotationInvariant(model_params)
        if self._use_spin:
            self._sf = SpinFeaturizer(model_params)

        self._ridge = Ridge(model_params.alpha)
        self._scaler = StandardScaler()

    def make_feature(
        self, structure_set: List[Structure], make_scaler: bool = False
    ) -> None:
        """Make the feature matrix from given structure set

        Args:
            structure_set (List[Structure]): list of structures
            make_scaler (bool): Whether to make scaler. Defaults to False.
        """
        x = self._ri(structure_set)

        if self._use_spin:
            spin_feature_matrix = self._sf(structure_set)
            x = np.hstack((x, spin_feature_matrix))

        if make_scaler:
            eid_end = len(structure_set)
            self._scaler = StandardScaler(with_mean=False).fit(x[:eid_end])

        self._x = self._scaler.transform(x, copy=False)

    @property
    def x(self) -> NDArray:
        """Return feature matrix

        Returns:
            NDArray: The feature matrix
        """
        return self._x

    @x.setter
    def x(self, new_feature) -> None:
        self._x = new_feature

    def train(self, train_index: List[int], y_kfold: NDArray) -> None:
        """Execute training of model

        Args:
            train_index (List[int]): The column id list of training matrix
            y_kfold (NDArray): The targets data in kfold dataset
        """
        self._ridge.fit(self._x[train_index], y_kfold[train_index])

    def predict(self, structure_set: List[Structure] = None) -> NDArray:
        """Predict total energy of given structures and forces on atoms in given structures

        Args:
            structure_set (List[Structure]): structure set

        Returns:
            NDArray: objective variable
        """
        if structure_set is not None:
            # Free memory by erasing feature matrix
            self._x = None

            self.make_feature(structure_set)

        return self._ridge.predict(self._x)

    def dump_model(self, model_dir: str) -> None:
        """Dump all the necessary data to restore the model

        Args:
            model_dir (str): Path to directory where files about the model are saved.
        """
        model_params_dict = self._ri.model_params.to_dict()  # type: ignore
        model_params_json_path = Path(model_dir) / "model_params.json"
        with model_params_json_path.open("w") as f:
            json.dump(model_params_dict, f, indent=4)

        ridge_file_path = Path(model_dir) / "ridge.pkl"
        with ridge_file_path.open("wb") as f:
            pickle.dump(self._ridge, f)

        scaler_file_path = Path(model_dir) / "scaler.pkl"
        with scaler_file_path.open("wb") as f:
            pickle.dump(self._scaler, f)


def load_model(model_dir: str) -> RILRM:
    """Load model object

    Args:
        model_dir (str): path to directory where files about the model are dumped

    Returns:
        RILRM: restored model object
    """
    model_params_json_path = Path(model_dir) / "model_params.json"
    with model_params_json_path.open("r") as f:
        model_params_dict = json.load(f)
    model_params = ModelParams.from_dict(model_params_dict)  # type: ignore

    model = RILRM(model_params)

    ridge_file_path = Path(model_dir) / "ridge.pkl"
    with ridge_file_path.open("rb") as f:
        ridge = pickle.load(f)
    model._ridge = ridge

    scaler_file_path = Path(model_dir) / "scaler.pkl"
    with scaler_file_path.open("rb") as f:
        scaler = pickle.load(f)
    model._scaler = scaler

    return model


def dump_model_as_lammps(model: RILRM, model_dir: str) -> None:
    """Dump model as LAMMPS potential file

    Args:
        model (RILRM): Trained model object
        model_dir (str): The directory where potential file for LAMMPS is saved.
    """
    content = make_content_of_lammps_file(model)

    lammps_file_path = Path(model_dir) / "mlp.lammps"
    with lammps_file_path.open("w") as f:
        f.write(content)


def make_content_of_lammps_file(model: RILRM) -> str:
    """Make content of potential file for LAMMPS

    Args:
        model (RILRM): Trained model object

    Returns:
        str: The content of potential file for LAMMPS
    """
    model_params = model._ri._model_params
    radial_params = model_params.make_radial_params()
    lines = []

    if model_params.composite_num == 1:
        elements_string = "Fe"
    else:
        elements_string = "Fe1 Fe2"
    lines.append(f"{elements_string} # element\n")
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
    lines.append(f"{len(radial_params)} # number of parameters\n")
    for item in radial_params:  # type: ignore
        lines.append(  # type: ignore
            f"{item[0]:15.15f} {item[1]:15.15f} # pair func. params\n"  # type: ignore
        )  # type: ignore
    mass = ["5.585000000000000e+01" for _ in range(model_params.composite_num)]
    mass_string = " ".join(mass)
    lines.append(mass_string)
    lines.append(" # atomic mass\n")
    lines.append("False # electrostatic\n")

    content = "".join(lines)

    return content
