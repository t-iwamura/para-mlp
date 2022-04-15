import copy
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

from para_mlp.config import Config
from para_mlp.data_structure import ModelParams
from para_mlp.model import RILRM
from para_mlp.utils import average, rmse

logger = logging.getLogger(__name__)


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


def make_param_grid(config: Config) -> Dict[str, Tuple[float]]:
    cutoff_radius_num = (
        int((config.cutoff_radius_max - config.cutoff_radius_min) / 2.0) + 1
    )
    cutoff_radius = tuple(
        radius
        for radius in np.linspace(
            config.cutoff_radius_min, config.cutoff_radius_max, cutoff_radius_num
        )
    )
    gaussian_params2_num = np.arange(10, config.gaussian_params2_num_max + 5, 5)

    param_grid = {
        "cutoff_radius": cutoff_radius,
        "gaussian_params2_num": gaussian_params2_num,
        "alpha": config.alpha,
    }

    return param_grid


def train_and_eval(
    config: Config,
    kfold_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
    yid_for_kfold: Dict[str, List[int]],
    yid_for_test: Dict[str, List[int]],
) -> Tuple[Any, ModelParams]:
    """Train candidate models and evaluate the best model of them

    Args:
        config (Config): configuration dataclass
        kfold_dataset (Dict[str, Any]): store energy, force, and structure set
        test_dataset (Dict[str, Any]): store energy, force, and structure set
        yid_for_kfold (Dict[str, List[int]]): The dataset to access yids for energy
            and force
        yid_for_test (Dict[str, List[int]]): The dataset to access yids for energy
            and force

    Returns:
        Tuple[Any, ModelParams]: model object and ModelParams dataclass
    """
    param_grid = make_param_grid(config)

    index_matrix = np.zeros(len(kfold_dataset["target"]))
    retained_model_rmse = 1e10

    for hyper_params in tqdm(ParameterGrid(param_grid)):

        model_params = ModelParams.from_dict(hyper_params)  # type: ignore

        model_params.gtinv_lmax = config.gtinv_lmax
        model_params.use_gtinv_sym = config.use_gtinv_sym
        model_params.use_force = config.use_force
        model_params.use_spin = config.use_spin

        model_params.set_api_params()
        model_params.make_feature_params()

        test_model = RILRM(model_params, kfold_dataset["structures"])

        logger.debug(" Test model")
        logger.debug("    params   : %s", hyper_params)
        logger.debug(f"    shape    : {test_model.x.shape}")

        test_model_rmses = []
        # test_model_rmses_energy, test_model_rmses_force = [], []
        kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=0)
        for train_index, valid_index in kf.split(index_matrix):
            test_model.train(train_index, kfold_dataset["target"])
            y_predict = test_model.predict()
            test_model_rmses.append(
                test_model.train_and_validate(
                    train_index, valid_index, kfold_dataset["target"]
                )
            )

        test_model_rmse = average(test_model_rmses)

        test_model_rmses = [round(rmse, 2) for rmse in test_model_rmses]
        logger.debug("    RMSE(valid)  : %s", test_model_rmses)
        logger.debug(f"    RMSE(average): {test_model_rmse}")

        if test_model_rmse < retained_model_rmse:
            retained_model_rmse = test_model_rmse
            retained_model = copy.deepcopy(test_model)
            retained_model_params = copy.deepcopy(hyper_params)

        logger.debug(" Retained model")
        logger.debug("    params   : %s", retained_model_params)
        logger.debug(f"    RMSE(val): {retained_model_rmse}")

    # Evaluate model's transferabilty for test data
    y_predict = retained_model.predict(test_dataset["structures"])

    model_score = rmse(y_predict, test_dataset["target"])
    logger.info(" Best model")
    logger.info("    params: %s", retained_model_params)
    logger.info(f"    RMSE(test): {model_score}")

    return retained_model, retained_model_params
