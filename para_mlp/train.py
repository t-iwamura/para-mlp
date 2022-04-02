import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold, ParameterGrid

from para_mlp.config import Config
from para_mlp.data_structure import ModelParams
from para_mlp.model import RILRM
from para_mlp.utils import rmse


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


def train_and_eval(
    config: Config,
    kfold_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
) -> Tuple[Any, ModelParams]:
    """Train candidate models and evaluate the best model of them

    Args:
        config (Config): configuration dataclass
        kfold_dataset (Dict[str, Any]): store energy, force, and structure set
        test_dataset (Dict[str, Any]): store energy, force, and structure set

    Returns:
        Tuple[Any, ModelParams]: model object and ModelParams dataclass
    """

    param_grid = {
        "alpha": config.alpha,
        "cutoff_radius": config.cutoff_radius,
    }

    index_matrix = np.zeros(len(kfold_dataset["target"]))
    retained_model_rmse = 1e10

    for hyper_params in ParameterGrid(param_grid):
        hyper_params["use_spin"] = config.use_spin

        model_params = ModelParams.from_dict(hyper_params)  # type: ignore
        model_params.make_feature_params()

        test_model = RILRM(model_params, kfold_dataset["structures"])

        kf = KFold(n_splits=10)
        test_model_rmse = 0.0
        for train_index, val_index in kf.split(index_matrix):
            test_model_rmse += test_model.train_and_validate(
                train_index, val_index, kfold_dataset["target"]
            )

        test_model_rmse = test_model_rmse / 10

        if test_model_rmse < retained_model_rmse:
            retained_model_rmse = test_model_rmse
            retained_model = copy.deepcopy(test_model)
            retained_model_params = copy.deepcopy(model_params)

    # Evaluate model's transferabilty for test data
    y_predict = retained_model.predict(test_dataset["structures"])

    model_score = rmse(y_predict, test_dataset["target"])
    print(f"Best model's score: {model_score}")

    return retained_model, retained_model_params
