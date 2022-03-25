import copy
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from dataclasses_json import dataclass_json


def make_model_params(hyper_params: dict):
    import mlpcpp  # type: ignore

    model_params = {}

    rotation_invariant = mlpcpp.Readgtinv(hyper_params["gtinv_order"], hyper_params["gtinv_lmax"], hyper_params["gtinv_sym"], 1)
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
    atomic_energy: float = -3.37689

    def make_feature_params(self):
        hyper_params = {}
        hyper_params["gaussian_params1"] = (1.0, 1.0, 1)
        hyper_params["gaussian_params2"] = (1.0, 5.0, 10)
        hyper_params["gtinv_order"] = 2
        hyper_params["gtinv_lmax"] = [3]
        self.lmax = copy.copy(hyper_params["gtinv_lmax"])[0]
        hyper_params["gtinv_sym"] = [False]

        model_params = make_model_params(hyper_params)

        self.lm_seq = model_params["lm_seq"]
        self.l_comb = model_params["l_comb"]
        self.lm_coeffs = model_params["lm_coeffs"]
        self.radial_params = model_params["radial_params"]
