import copy
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Tuple

import numpy as np
from dataclasses_json import dataclass_json


def make_model_params(hyper_params: dict) -> dict:
    import mlpcpp  # type: ignore

    model_params = {}

    rotation_invariant = mlpcpp.Readgtinv(
        hyper_params["gtinv_order"],
        hyper_params["gtinv_lmax"],
        hyper_params["gtinv_sym"],
        1,
    )
    model_params["lm_seq"] = rotation_invariant.get_lm_seq()
    model_params["l_comb"] = rotation_invariant.get_l_comb()
    model_params["lm_coeffs"] = rotation_invariant.get_lm_coeffs()

    radial_params = hyper_params["gaussian_params1"]
    radial_params1 = np.linspace(radial_params[0], radial_params[1], radial_params[2])
    radial_params = hyper_params["gaussian_params2"]
    radial_params2 = np.linspace(radial_params[0], radial_params[1], radial_params[2])
    model_params["radial_params"] = list(product(radial_params1, radial_params2))

    return model_params


@dataclass_json
@dataclass
class ModelParams:
    # what data to use
    use_force: bool = True
    use_stress: bool = False
    atomic_energy: float = -3.37689
    # functional form
    composite_num: int = 1
    polynomial_model: int = 1
    polynomial_max_order: int = 2
    # feature settings
    # API params
    feature_type: str = "gtinv"
    cutoff_radius: float = 6.0
    radial_func: str = "gaussian"
    gaussian_params1: Tuple[float, float, int] = (1.0, 1.0, 1)
    gaussian_params2: Tuple[float, float, int] = (1.0, 5.0, 10)
    gtinv_order: float = 2
    gtinv_lmax: Tuple[float] = (3,)
    gtinv_sym: Tuple[bool] = (False,)
    # naive params
    lmax: Any = None
    lm_seq: Any = None
    l_comb: Any = None
    lm_coeffs: Any = None
    radial_params: Any = None
    # misc
    alpha: float = None

    def make_feature_params(self) -> None:
        hyper_params: Dict[str, Any] = {
            "gaussian_params1": list(self.gaussian_params1),
            "gaussian_params2": list(self.gaussian_params2),
            "gtinv_order": self.gtinv_order,
            "gtinv_lmax": [i for i in self.gtinv_lmax],
            "gtinv_sym": [i for i in self.gtinv_sym],
        }
        new_model_params = make_model_params(hyper_params)

        self.lmax = copy.deepcopy(hyper_params["gtinv_lmax"])[0]

        self.lm_seq = new_model_params["lm_seq"]
        self.l_comb = new_model_params["l_comb"]
        self.lm_coeffs = new_model_params["lm_coeffs"]
        self.radial_params = new_model_params["radial_params"]
