import copy
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant


def dump_model(model: Any, model_params: ModelParams, model_dir: str) -> None:
    model_filepath = Path(".") / model_dir / "model.pkl"
    with model_filepath.open("wb") as f:
        pickle.dump((model, model_params), f)


def load_model(model_dir: str):
    model_filepath = Path(".") / model_dir / "model.pkl"
    with model_filepath.open("rb") as f:
        model, model_params = pickle.load(f)

    return model, model_params


def rmse(y_predict: NDArray, y_target: NDArray) -> float:
    return np.sqrt(np.mean(np.square(y_predict - y_target)))


def train_and_eval(
    model_params: ModelParams,
    kfold_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
) -> Tuple[Any, ModelParams]:

    cutoff_radius = [6.0, 7.0, 8.0]

    test_model = Ridge(alpha=1e-2)

    retained_model_rmse = 1e10
    for val in cutoff_radius:
        model_params.cutoff_radius = val
        ri = RotationInvariant(kfold_dataset["structures"], model_params)

        x = ri.x

        kf = KFold(n_splits=10)
        test_model_rmse = 0.0
        for train_index, val_index in kf.split(x):
            test_model.fit(x[train_index], kfold_dataset["energy"][train_index])

            y_predict = test_model.predict(x[val_index])
            y_target = kfold_dataset["energy"][val_index]
            test_model_rmse += rmse(y_predict, y_target)

        test_model_rmse = test_model_rmse / 10

        if test_model_rmse < retained_model_rmse:
            retained_model_rmse = test_model_rmse
            retained_model = copy.deepcopy(test_model)
            retained_model_params = copy.deepcopy(model_params)

    # Evaluate model's transferabilty for test data
    ri = RotationInvariant(test_dataset["structures"], retained_model_params)
    x_test = ri.x

    y_predict = retained_model.predict(x_test)
    model_score = rmse(y_predict, test_dataset["energy"])
    print(f"Best model's score: {model_score}")

    return retained_model, retained_model_params
