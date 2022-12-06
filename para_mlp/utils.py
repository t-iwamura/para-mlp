import copy
import json
import site
from itertools import product
from math import floor, log10
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from para_mlp.config import Config


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


class SampleWeightCalculator:
    def __init__(
        self, config: Config, yids_for_kfold: Dict[str, List[int]], n_structure: int
    ) -> None:
        self._yids_for_kfold = yids_for_kfold
        self._n_structure = n_structure
        self._use_force = config.use_force

        self._energy_weight = config.energy_weight
        self._force_weight = config.force_weight
        self._high_energy_weight = config.high_energy_weight

        if self._high_energy_weight != 1.0:
            high_energy_structures_path = (
                Path(config.model_dir) / "high_energy_structures"
            )
            with high_energy_structures_path.open("r") as f:
                self._high_energy_structure_id = [int(line.strip()) - 1 for line in f]

    def arrange_high_energy_index(self, force_id_unit: int) -> None:
        """Arrange self._high_energy_index

        Args:
            force_id_unit (int): The length of force ids per one structure
        """
        if self._high_energy_weight != 1.0:
            yids_for_high_energy = make_yids_for_structure_ids(
                self._high_energy_structure_id,
                self._n_structure,
                force_id_unit,
                self._use_force,
            )
            self._high_energy_index = [
                i
                for i, yid in enumerate(self._yids_for_kfold["target"])
                if yid in yids_for_high_energy["target"]
            ]

    def make_sample_weight(
        self, yids_for_train: List[int], n_energy_data: int
    ) -> NDArray:
        """Make sample_weight for training

        Args:
            yids_for_train (List[int]): The yids of kfold target data for training
            n_energy_data (int): The number of energy data for training

        Returns:
            NDArray: sample_weight
        """
        if (
            (self._energy_weight == 1.0)
            and (self._force_weight == 1.0)
            and (self._high_energy_weight == 1.0)
        ):
            return None

        self._sample_weight = np.ones(shape=(len(yids_for_train),))

        if self._energy_weight != 1.0:
            self._sample_weight[:n_energy_data] *= self._energy_weight

        if self._force_weight != 1.0:
            self._sample_weight[n_energy_data:] *= self._force_weight

        if self._high_energy_weight != 1.0:
            high_energy_yids = [
                i
                for i, yid in enumerate(yids_for_train)
                if yid in self._high_energy_index
            ]
            self._sample_weight[high_energy_yids] *= self._high_energy_weight

        return self._sample_weight


def get_head_commit_id() -> str:
    """Get the commit id of HEAD of this repo

    Returns:
        str: the commit id of HEAD
    """
    git_root_path = Path(__file__).resolve().parent.parent / ".git"
    head_path = git_root_path / "HEAD"
    with head_path.open("r") as f:
        head_info_file = f.readline().split(":")[-1].strip()

    head_info_file_path = git_root_path / head_info_file
    with head_info_file_path.open("r") as f:
        commit_id = f.readline().strip()

    return commit_id


def dump_version_info(model_dir: str) -> None:
    """Dump json file about the versions of mlpcpp and para-mlp

    Args:
        model_dir (str): the path to model directory
    """
    mlpcpp_version_info_path = (
        Path(site.getsitepackages()[0]) / "mlpcpp_version_info.json"
    )
    with mlpcpp_version_info_path.open("r") as f:
        mlpcpp_version_dict = json.load(f)
    version_info_dict = {
        "mlpcpp_version": mlpcpp_version_dict["version"],
        "para_mlp_version": get_head_commit_id(),
    }

    version_info_path = Path(model_dir) / "version_info.json"
    with version_info_path.open("w") as f:
        json.dump(version_info_dict, f, indent=4)
