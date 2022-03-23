from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelParams:
    # what data to use
    use_force: bool = False
    use_stress: bool = False
    # functional form
    composite_num: int = 1
    polynomial_model: int = 1
    polynomial_max_order: int = 2
    # feature settings
    feature_type: str = "gtinv"
    cutoff_radius: float = 6.0
    lmax: Any = None
    lm_seq: Any = None
    l_comb: Any = None
    lm_coeffs: Any = None
    radial_func: str = "gaussian"
    radial_params: Any = None
    atomic_energy: float = None
