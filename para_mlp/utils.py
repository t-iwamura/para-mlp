import copy
from itertools import product
from math import floor, log10
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray


def round_to_4(x: float) -> float:
    """Round x to 4 significant figures

    Args:
        x (float): input value

    Returns:
        float: rounded value
    """
    return round(x, -int(floor(log10(abs(x)))) + 3)


def average(sequence: Sequence) -> float:
    """Calculate average of given sequence

    Args:
        sequence (Sequence): List, Tuple, etc ...

    Returns:
        float: average
    """
    return sum(sequence) / len(sequence)


def rmse(y_predict: NDArray, y_target: NDArray) -> float:
    """Calculate RMSE of objective variable

    Args:
        y_predict (NDArray): predicted objective variable
        y_target (NDArray): target objected variable

    Returns:
        float: RMSE
    """
    return np.sqrt(np.mean(np.square(y_predict - y_target)))


def make_yids_for_structure_ids(
    structure_id: List[int],
    energy_id_length: int,
    force_id_unit: int,
    use_force: bool = False,
) -> Dict[str, List[int]]:
    """Make yids for given structure ids

    Args:
        structure_id (List[int]): The ids of chosen structures
        energy_id_length (int): The length of columns where energy ids are set
        force_id_unit (int): The length of force ids per one structure
        use_force (bool, optional): Whether to use force. Defaults to False.

    Returns:
        Dict[str, List[int]]: The yids for given structures. The keys are 'energy',
            'force', and 'target'.
    """
    yids_dict = {}

    # Calculate yids for energy data
    yids_dict["energy"] = copy.deepcopy(structure_id)

    if use_force:
        # Calculate yids for force data
        yids_dict["force"] = [
            energy_id_length + force_id_unit * sid + force_id
            for sid, force_id in product(structure_id, range(force_id_unit))
        ]
        yids_dict["target"] = [*yids_dict["energy"], *yids_dict["force"]]
    else:
        yids_dict["target"] = copy.deepcopy(yids_dict["energy"])

    return yids_dict
