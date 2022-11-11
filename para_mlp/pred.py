import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from para_mlp.model import load_model
from para_mlp.preprocess import create_dataset, load_ids_for_test_and_kfold

logger = logging.getLogger(__name__)


def predict_property(
    model_dir: str,
    structure_file: str,
    types_list: List[List[int]] = None,
    use_force: bool = True,
) -> Dict[str, Any]:
    """Predict cohesive energy and forces on atoms of given structure

    Args:
        model_dir (str): The path to model directory where all the necessary files
            are saved.
        structure_file (str): The path to structure.json.
        types_list (List[List[int]], optional): List of element types about
            given structure. Defaults to None.
        use_force (bool, optional): Whether to use force data or not.
            Defaults to True.

    Returns:
        Dict[str, Any]: The dict of cohesive energy, forces on atoms, and
            calculation time. The keys are "energy", "force", and "calc_time".
            The shape of force is as follows.
                shape=(3 * {n_atoms_in_structure}, 1)
    """
    with open(structure_file) as f:
        structure_dict = json.load(f)
    structure = Structure.from_dict(structure_dict)

    # Start prediction
    predict_dict = {}
    start = time.time()

    model = load_model(model_dir)
    model._ri.model_params.use_force = use_force
    y = model.predict([structure], types_list=types_list)

    predict_dict["energy"] = y[0]
    if use_force:
        predict_dict["force"] = y[1:]

    end = time.time()
    predict_dict["calc_time"] = end - start

    return predict_dict


def evaluate_energy_prediction_for_dataset(model_dir: str) -> None:
    """Evaluate energy prediction accuracy of given model for kfold and test dataset

    Args:
        model_dir (str): path to model directory
    """
    # Arrange structures and energy in overall dataset
    logger.info(" Arrange structures and energy in overall dataset")

    data_dir_path = Path.home() / "para-mlp" / "data"
    targets_json_path = Path.home() / "para-mlp" / "configs" / "targets.json"
    dataset = create_dataset(str(data_dir_path), targets_json=str(targets_json_path))

    # Obtain structures and energy for kfold and test dataset
    logger.info(" Split dataset")

    processing_dir_path = Path.home() / "para-mlp" / "data" / "processing"
    structure_id, yids_for_kfold, yids_for_test = load_ids_for_test_and_kfold(
        processing_dir=str(processing_dir_path), use_force=True
    )
    kfold_dataset = {
        "structures": [dataset["structures"][sid] for sid in structure_id["kfold"]],
        "energy": dataset["target"][yids_for_kfold["energy"]],
    }
    test_dataset = {
        "structures": [dataset["structures"][sid] for sid in structure_id["test"]],
        "energy": dataset["target"][yids_for_test["energy"]],
    }

    logger.info(" Load model")

    model = load_model(model_dir)
    n_atoms_in_structure = len(kfold_dataset["structures"][0].sites)

    # Evaluate prediction accuracy for kfold data
    logger.info(" Predict energy for kfold dataset")

    y = model.predict(kfold_dataset["structures"])

    energy_id_end = len(kfold_dataset["structures"])
    energy_predict = y[:energy_id_end] / n_atoms_in_structure
    energy_expected = kfold_dataset["energy"] / n_atoms_in_structure

    logger.info(" Record energy prediction accuracy")

    kfold_energy_filename = "/".join([model_dir, "prediction", "kfold_energy.out"])
    record_energy_prediction_accuracy(
        energy_predict, energy_expected, output_filename=kfold_energy_filename
    )

    # Evaluate prediction accuracy for test data
    logger.info(" Predict energy for test dataset")

    y = model.predict(test_dataset["structures"])

    energy_id_end = len(test_dataset["structures"])
    energy_predict = y[:energy_id_end] / n_atoms_in_structure
    energy_expected = test_dataset["energy"] / n_atoms_in_structure

    logger.info(" Record energy prediction accuracy")

    test_energy_filename = "/".join([model_dir, "prediction", "test_energy.out"])
    record_energy_prediction_accuracy(
        energy_predict, energy_expected, output_filename=test_energy_filename
    )


def record_energy_prediction_accuracy(
    predict: NDArray, expected: NDArray, output_filename: str
) -> None:
    """Record energy prediction accuracy of the model

    Args:
        predict (NDArray): predicted energy for given dataset
        expected (NDArray): DFT energy for given dataset
        output_filename (str): path to output file
    """
    output_dir_path = Path(output_filename).parent
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    fmt = ("%5.10f", "%5.10f", "%.10e")
    header = "MLP, DFT, MLP-DFT"

    prediction_results = np.stack((predict, expected, predict - expected), axis=1)
    np.savetxt(output_filename, prediction_results, fmt=fmt, header=header)
