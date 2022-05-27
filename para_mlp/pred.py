import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray
from pymatgen.core import Structure

from para_mlp.model import load_model


def predict_property(model_dir: str, structure_file: str) -> Dict[str, Any]:
    """Predict cohesive energy and forces on atoms of given structure

    Args:
        model_dir (str): The path to model directory where all the necessary files
            are saved.
        structure_file (str): The path to structure.json

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
    y = model.predict([structure])

    predict_dict["energy"] = y[0]
    predict_dict["force"] = y[1:]

    end = time.time()
    predict_dict["calc_time"] = end - start

    return predict_dict


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
