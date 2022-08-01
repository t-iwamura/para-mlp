import copy
from dataclasses import dataclass
from itertools import product
from typing import List, Tuple

import numpy as np
from dataclasses_json import dataclass_json


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
    polynomial_max_order: int = 1
    # feature settings
    # API params
    # rotation invariant settings
    feature_type: str = "gtinv"
    cutoff_radius: float = 6.0
    radial_func: str = "gaussian"
    gaussian_params2_flag: int = 1
    gaussian_params1: Tuple[float, float, int] = (1.0, 1.0, 1)
    gaussian_params2: Tuple[float, float, int] = (0.0, 6.0, 5)
    gaussian_params2_num: int = 5
    gtinv_order: int = 2
    gtinv_lmax: Tuple[int, ...] = (3,)
    lmax: int = 3
    use_gtinv_sym: bool = False
    gtinv_sym: Tuple[bool, ...] = (False,)
    # spin feature settings
    use_spin: bool = False
    magnetic_cutoff_radius: float = 5
    coeff_order_max: int = 3
    # misc
    alpha: float = 1e-2

    def set_api_params(self) -> None:
        if self.gaussian_params2_flag == 1:
            gaussian_center_end = self.cutoff_radius - 1.0
            self.gaussian_params2 = (
                0.0,
                self.cutoff_radius - 1.0,
                self.gaussian_params2_num,
            )
        elif self.gaussian_params2_flag == 2:
            gaussian_center_end = (
                self.cutoff_radius
                * (self.gaussian_params2_num - 1)
                / self.gaussian_params2_num
            )
            self.gaussian_params2 = (
                0.0,
                gaussian_center_end,
                self.gaussian_params2_num,
            )
        if self.feature_type == "gtinv":
            self.gtinv_order = len(self.gtinv_lmax) + 1
            self.lmax = copy.copy(self.gtinv_lmax[0])
            self.gtinv_sym = tuple(
                self.use_gtinv_sym for i in range(self.gtinv_order - 1)
            )
        else:
            self.lmax = 0

    def make_radial_params(self) -> List[Tuple[float, float]]:
        """Make radial parameters

        Returns:
            List[Tuple[float, float]]: list of gaussian parameter pair
        """
        gaussian_params1 = list(self.gaussian_params1)
        gaussian_params2 = list(self.gaussian_params2)

        radial_params1 = np.linspace(
            gaussian_params1[0], gaussian_params1[1], gaussian_params1[2]
        )
        radial_params2 = np.linspace(
            gaussian_params2[0], gaussian_params2[1], gaussian_params2[2]
        )

        return list(product(radial_params1, radial_params2))

    def make_feature_params(self) -> dict:
        """Make feature parameters

        Returns:
            dict: dict of feature parameters
        """
        feature_params = {}

        import mlpcpp  # type: ignore

        if self.feature_type == "gtinv":
            feature_coeff_maker = mlpcpp.Readgtinv(
                self.gtinv_order,
                list(self.gtinv_lmax),
                list(self.gtinv_sym),
                self.composite_num,
            )
            feature_params["lm_seq"] = feature_coeff_maker.get_lm_seq()
            feature_params["l_comb"] = feature_coeff_maker.get_l_comb()
            feature_params["lm_coeffs"] = feature_coeff_maker.get_lm_coeffs()
        else:
            feature_params["lm_seq"] = []
            feature_params["l_comb"] = []
            feature_params["lm_coeffs"] = []

        feature_params["radial_params"] = self.make_radial_params()

        return feature_params
