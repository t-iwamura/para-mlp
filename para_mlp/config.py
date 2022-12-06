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
    composite_num: int = 1
    polynomial_model: int = 1
    polynomial_max_order: int = 1
    is_paramagnetic: bool = False
    delta_learning: bool = False
    cutoff_radius_min: float = 6.0
    cutoff_radius_max: float = 12.0
    gaussian_params2_flag: int = 1
    gaussian_params2_num_min: int = 10
    gaussian_params2_num_max: int = 10
    feature_type: str = "gtinv"
    gtinv_lmax: Tuple[int, ...] = (9, 7, 3, 2, 2)
    use_gtinv_sym: bool = False
    use_spin: bool = False
    # preprocessing
    use_force: bool = False
    shuffle: bool = True
    use_cache_to_split_data: bool = True
    # training
    alpha: Tuple[float, ...] = (1e-2, 1e-3)
    energy_weight: float = 1.0
    force_weight: float = 1.0
    high_energy_weight: float = 1.0
    n_splits: int = 5
    metric: str = "energy"
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
