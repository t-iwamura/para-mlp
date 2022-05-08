import json
import time
from typing import Any, Dict

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
