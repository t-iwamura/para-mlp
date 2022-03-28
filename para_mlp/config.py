from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    # dataset path
    data_dir: str = "data"
    targets_json: str = "configs/targets.json"
    # model parameter space
    # preprocessing
    shuffle: bool = True
    # training
    # misc
    model_dir: str = "models"
