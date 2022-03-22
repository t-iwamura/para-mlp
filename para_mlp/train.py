from para_mlp.data_structure import ModelParams
from para_mlp.preprocess import make_model_params


def train():
    # cutoff_radius = [6.0, 7.0, 8.0]
    model_params = {}
    model_params["use_force"] = False
    model_params["use_stress"] = False
    model_params["polynomial_model"] = 1
    model_params["polynomial_max_order"] = 2
    model_params["radial_func"] = "gaussian"
    model_params["feature_type"] = "gtinv"
    model_params["composite_num"] = 1
    model_params["cutoff_radius"] = 6.0
    model_params["atomic_energy"] = -3.37689
    hyper_params = {}
    hyper_params["gaussian_params1"] = (1.0, 1.0, 1)
    hyper_params["gaussian_params2"] = (1.0, 5.0, 10)
    hyper_params["gtinv_order"] = 2
    hyper_params["gtinv_lmax"] = 3
    hyper_params["gtinv_sym"] = 0
    model_params["lmax"] = hyper_params["gtinv_lmax"]

    model_params = ModelParams.from_dict(make_model_params(hyper_params, model_params))

    # TODO: kf = KFold(n_splits=10)
