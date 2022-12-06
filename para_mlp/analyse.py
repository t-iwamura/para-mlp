import copy
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pygmo as pg
from tqdm import tqdm

logger = logging.getLogger(__name__)


def parse_std_log(logfile: str) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Parse std.log

    Args:
        logfile (str): The path to std.log

    Returns:
        Tuple[List[Dict[str, Any]], List[float]]: The model parameters of all the test
            models and scores of the models
    """
    logfile_path = Path(logfile)
    with logfile_path.open("r") as f:
        lines = f.readlines()

    # Start parsing
    block_column_ids = [i for i, line in enumerate(lines) if "Test model" in line]

    models, scores = [], []
    for column_id in block_column_ids:
        model_params = {}
        model_params_list = re.findall(r"'[\w]+': [0-9.]+", lines[column_id + 1])
        key, val = model_params_list[0].split(": ")
        model_params[key.replace("'", "")] = float(val)
        key, val = model_params_list[1].split(": ")
        model_params[key.replace("'", "")] = float(val)
        key, val = model_params_list[2].split(": ")
        model_params[key.replace("'", "")] = int(val)
        models.append(model_params)

        score_match = re.search(r"[0-9.]+\n", lines[column_id + 5])
        score = float(score_match.group().strip())
        scores.append(score)

    return models, scores


def search_pareto_optimal(search_dir: str, metric: str = "energy") -> Dict[str, Any]:
    """Search pareto optimal potentials

    Args:
        search_dir (str): path to searching directory
        metric (str, optional): The metric for searching pareto optimal potentials.
            Defaults to "energy".

    Returns:
        Dict[str, Any]: The dict about calculation details
    """
    model_names = []
    rmse_energies, rmse_forces, calc_times = [], [], []

    calc_info_dict: Dict[str, Any] = {"search_dir": search_dir, "metric": metric}
    all_models_dict, pareto_optimal_dict = {}, {}

    # Define matching objects
    rmse_energy_pattern = re.compile(r"RMSE\(test, energy, meV/atom\):\s+([\d.]+)")
    rmse_force_pattern = re.compile(r"RMSE\(test, force, eV/ang\):\s+([\d.]+)")

    logger.info(" Searching log directory")

    log_dir_list = [
        log_dir_path.parent
        for log_dir_path in Path(search_dir).glob("**/[0-9][0-9][0-9]/predict.json")
    ]
    for log_dir_path in tqdm(log_dir_list):
        model_name = str(log_dir_path)
        model_names.append(model_name)

        property_dict = {}
        std_log_json_path = log_dir_path / "model.log"
        f = std_log_json_path.open("r")
        if (metric == "energy") or (metric == "energy_and_force"):
            for line in iter(f.readline, ""):
                m = rmse_energy_pattern.search(line)
                if m is not None:
                    rmse_energy = float(m.group(1))
                    rmse_energies.append(rmse_energy)
                    property_dict["rmse_energy"] = rmse_energy
                    break

        if (metric == "force") or (metric == "energy_and_force"):
            for line in iter(f.readline, ""):
                m = rmse_force_pattern.search(line)
                if m is not None:
                    rmse_force = float(m.group(1))
                    rmse_forces.append(rmse_force)
                    property_dict["rmse_force"] = rmse_force
                    break
        f.close()

        pred_json_path = log_dir_path / "predict.json"
        with pred_json_path.open("r") as f:
            pred_dict = json.load(f)
        calc_times.append(pred_dict["calc_time"])
        property_dict["calc_time"] = pred_dict["calc_time"]

        all_models_dict[model_name] = property_dict

    all_models_dict = dict(
        sorted(all_models_dict.items(), key=lambda item: item[1]["rmse_energy"])
    )
    calc_info_dict["all"] = all_models_dict

    rmse_energies = np.array(rmse_energies).reshape((-1, 1))
    rmse_forces = np.array(rmse_forces).reshape((-1, 1))
    calc_times = np.array(calc_times).reshape((-1, 1))

    if metric == "energy":
        score_array = np.hstack((rmse_energies, calc_times))
    elif metric == "force":
        score_array = np.hstack((rmse_forces, calc_times))
    elif metric == "energy_and_force":
        score_array = np.hstack((rmse_energies, rmse_forces, calc_times))

    non_dominated_frontiers, _, _, _ = pg.fast_non_dominated_sorting(score_array)

    for pareto_id in non_dominated_frontiers[0]:
        pareto_optimal_dict[model_names[pareto_id]] = copy.deepcopy(
            all_models_dict[model_names[pareto_id]]
        )

    pareto_optimal_dict = dict(
        sorted(pareto_optimal_dict.items(), key=lambda item: item[1]["rmse_energy"])
    )
    calc_info_dict["pareto"] = pareto_optimal_dict

    return calc_info_dict


def find_best_model_in_metric(
    pareto_property_dict: Dict[str, Dict[str, float]], time_ratio: int = 1
) -> str:
    """Find best model with regard to the metric, {time_ratio} * t + dE

    Args:
        pareto_property_dict (Dict[str, Dict[str, float]]): property dict about
            pareto optimal potentials
        time_ratio (int, optional): ratio in the metric formula. Defaults to 1.0.

    Returns:
        str: best model name
    """
    scores_in_metric = {
        model_name: time_ratio * property_dict["calc_time"] * 1e3
        + property_dict["rmse_energy"]
        for model_name, property_dict in pareto_property_dict.items()
    }
    best_model = min(scores_in_metric, key=scores_in_metric.get)
    return best_model
