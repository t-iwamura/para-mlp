import copy
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant
from para_mlp.preprocess import create_dataset, split_dataset


def dump_model(model: Any, model_params: ModelParams, model_dir: str) -> None:
    dump_filepath = Path("/".join([model_dir, "model.pkl"]))
    with dump_filepath.open("wb") as f:
        pickle.dump((model, model_params), f)


def rmse(y_predict: NDArray, y_target: NDArray) -> float:
    return np.sqrt(np.mean(np.square(y_predict - y_target)))


def train() -> None:
    dataset = create_dataset()
    structure_train, structure_test, y_train, y_test = split_dataset(dataset)

    model_params = ModelParams()
    model_params.make_feature_params()

    cutoff_radius = [6.0, 7.0, 8.0]

    test_model = Ridge(alpha=1e-2)

    retained_model_rmse = 1e10
    for val in cutoff_radius:
        model_params.cutoff_radius = val

        feature_generator = RotationInvariant(structure_train, model_params)
        x_train = feature_generator.x

        test_rmse = 0.0
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(x_train):
            test_model.fit(x_train[train_index], y_train[train_index])

            y_predict = test_model.predict(x_train[test_index])
            y_target = y_train[test_index]
            test_rmse += rmse(y_predict, y_target)

        test_rmse = test_rmse / 10

        if test_rmse < retained_model_rmse:
            retained_model_rmse = test_rmse
            retained_model = copy.deepcopy(test_model)
            retained_model_params = copy.deepcopy(model_params)

    # Evaluate model's transferabilty for test data
    feature_generator = RotationInvariant(structure_test, retained_model_params)
    x_test = feature_generator.x

    y_predict = retained_model.predict(x_test)
    model_score = rmse(y_predict, y_test)
    print("Best model's score: {}".format(model_score))

    model_dir = "models"
    dump_model(retained_model, retained_model_params, model_dir)
