import copy
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm

from para_mlp.config import Config
from para_mlp.data_structure import ModelParams
from para_mlp.model import RILRM
from para_mlp.utils import average, make_yids_for_structure_ids, rmse

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
    """Make parameter grid for grid search

    Args:
        config (Config): config to make machine learning model

    Returns:
        Dict[str, Tuple[float]]: The parameter grid. All the possible values are stored
            for each key.
    """
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
) -> Tuple[Any, ModelParams]:
    """Train candidate models and evaluate the best model of them

    Args:
        config (Config): configuration dataclass
        kfold_dataset (Dict[str, Any]): store energy, force, and structure set
        test_dataset (Dict[str, Any]): store energy, force, and structure set

    Returns:
        Tuple[Any, ModelParams]: model object and ModelParams dataclass
    """
    param_grid = make_param_grid(config)

    n_kfold_structure = len(kfold_dataset["structures"])
    index_matrix = np.zeros(n_kfold_structure)
    force_id_unit = (kfold_dataset["target"].shape[0] // n_kfold_structure) - 1
    retained_model_rmse = 1e10

    for hyper_params in tqdm(ParameterGrid(param_grid)):

        # Keep hyper_params to store variable parameters
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

        test_model_rmses_energy, test_model_rmses_force = [], []
        kf = KFold(n_splits=config.n_splits, shuffle=True, random_state=0)
        for train_index, valid_index in kf.split(index_matrix):
            yids_for_train = make_yids_for_structure_ids(
                train_index, n_kfold_structure, force_id_unit, config.use_force
            )
            yids_for_valid = make_yids_for_structure_ids(
                valid_index, n_kfold_structure, force_id_unit, config.use_force
            )
            test_model.train(yids_for_train["target"], kfold_dataset["target"])

            y_predict = test_model.predict()
            test_model_rmses_energy.append(
                rmse(
                    y_predict[yids_for_valid["energy"]],
                    kfold_dataset["target"][yids_for_valid["energy"]],
                )
            )
            if config.use_force:
                test_model_rmses_force.append(
                    rmse(
                        y_predict[yids_for_valid["force"]],
                        kfold_dataset["target"][yids_for_valid["force"]],
                    )
                )

        test_model_rmse_energy = average(test_model_rmses_energy) * 1e3
        test_model_rmses_energy = [
            round(rmse, 2) * 1e3 for rmse in test_model_rmses_energy
        ]
        logger.debug("    RMSE(energy, meV/atom)  : %s", test_model_rmses_energy)
        logger.debug(f"    RMSE(energy, average, meV/atom): {test_model_rmse_energy}")

        if config.use_force:
            test_model_rmse_force = average(test_model_rmses_force)
            test_model_rmses_force = [round(rmse, 2) for rmse in test_model_rmses_force]
            logger.debug("    RMSE(force, eV/ang)   : %s", test_model_rmses_force)
            logger.debug(f"    RMSE(force, average, eV/ang) : {test_model_rmse_force}")

        if config.metric == "energy":
            test_model_rmse = test_model_rmse_energy
            rmse_description = "energy, meV/atom"
        elif config.use_force and (config.metric == "force"):
            test_model_rmse = test_model_rmse_force
            rmse_description = "force, eV/ang"
        else:
            print("Cannot use RMSE(force) as metric because force data is not used.")
            sys.exit(1)

        if test_model_rmse < retained_model_rmse:
            retained_model_rmse = test_model_rmse
            retained_model = copy.deepcopy(test_model)
            retained_model_params = copy.deepcopy(hyper_params)

        logger.debug(" Retained model")
        logger.debug("    params   : %s", retained_model_params)
        logger.debug(f"    RMSE({rmse_description: <16}): {retained_model_rmse}")

    # Evaluate model's transferabilty for test data
    y_predict = retained_model.predict(test_dataset["structures"])

    energy_id_end = len(test_dataset["structures"])
    model_score_energy = (
        rmse(y_predict[:energy_id_end], test_dataset["target"][:energy_id_end]) * 1e3
    )
    model_score_force = rmse(
        y_predict[energy_id_end:], test_dataset["target"][energy_id_end:]
    )
    logger.info(" Best model")
    logger.info("    params: %s", retained_model_params)
    logger.info(f"    RMSE(test, energy, meV/atom): {model_score_energy}")
    logger.info(f"    RMSE(test, force, eV/ang): {model_score_force}")

    return retained_model, retained_model_params
