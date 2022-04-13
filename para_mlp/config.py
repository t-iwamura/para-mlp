import json
from dataclasses import dataclass
from typing import Tuple

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    # dataset path
    data_dir: str = "data"
    targets_json: str = "configs/targets.json"
    # model parameter space
    cutoff_radius_min: float = 6.0
    cutoff_radius_max: float = 12.0
    gaussian_params2_num_max: int = 10
    gtinv_lmax: Tuple[int, ...] = (9, 7, 3, 2, 2)
    use_gtinv_sym: bool = False
    use_spin: bool = False
    # preprocessing
    use_force: bool = False
    shuffle: bool = True
    use_cache_to_split_data: bool = True
    # training
    alpha: Tuple[float, ...] = (1e-2, 1e-3)
    # misc
    save_log: bool = False
    model_dir: str = "models"
    n_jobs: int = -1


def load_config(path: str) -> Config:
    """Load configs/*.json

    Args:
        path (str): path to configs/*.json

    Returns:
        Config: training configuration dataclass
    """
    with open(path, "r") as f:
        config_dict = json.load(f)
    return Config.from_dict(config_dict)  # type: ignore
