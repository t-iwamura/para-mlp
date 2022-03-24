from copy import copy
from dataclasses import dataclass
from typing import Any

from dataclasses_json import dataclass_json

from para_mlp.preprocess import make_model_params


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
        hyper_params["gtinv_sym"] = [False]

        model_params = make_model_params(hyper_params)
        model_params = ModelParams.from_dict(model_params)

        self.lmax = copy.copy(hyper_params["gtinv_lmax"])[0]
        self.lm_seq = model_params["lm_seq"]
        self.l_comb = model_params["l_comb"]
        self.lm_coeffs = model_params["lm_coeffs"]
        self.radial_params = model_params["radial_params"]
