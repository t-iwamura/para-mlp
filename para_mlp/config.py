from dataclasses import dataclass
from typing import Tuple

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class Config:
    # model parameter space
    # preprocessing
    structure_ids: Tuple[str, ...] = tuple((str(i + 1).zfill(5) for i in range(100)))
    shuffle: bool = True
    # training
    # misc
