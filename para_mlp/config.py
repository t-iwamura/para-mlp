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
    cutoff_radius: Tuple[float, ...] = (6.0, 7.0)
    # preprocessing
    use_force: bool = True
    shuffle: bool = True
    # training
    alpha: Tuple[float, ...] = (1e-2, 1e-3)
    # misc
    model_dir: str = "models"
    n_jobs: int = 1
