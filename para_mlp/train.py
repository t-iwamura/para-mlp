import copy

from para_mlp.data_structure import ModelParams
from para_mlp.preprocess import create_dataset, make_model_params, split_dataset


def train():
    # cutoff_radius = [6.0, 7.0, 8.0]
    model_params = {}
    model_params["use_force"] = False
    model_params["use_stress"] = False
    model_params["composite_num"] = 1
    model_params["polynomial_model"] = 1
    model_params["polynomial_max_order"] = 2
    model_params["feature_type"] = "gtinv"
    model_params["cutoff_radius"] = 6.0
    model_params["radial_func"] = "gaussian"
    model_params["atomic_energy"] = -3.37689

    hyper_params = {}
    hyper_params["gaussian_params1"] = (1.0, 1.0, 1)
    hyper_params["gaussian_params2"] = (1.0, 5.0, 10)
    hyper_params["gtinv_order"] = 2
    hyper_params["gtinv_lmax"] = [3]
    hyper_params["gtinv_sym"] = [False]
    model_params["lmax"] = copy.copy(hyper_params["gtinv_lmax"])[0]

    model_params.update(make_model_params(hyper_params))
    model_params = ModelParams.from_dict(model_params)

    structure_ids = (str(i + 1).zfill(5) for i in range(10))
    dataset = create_dataset(structure_ids)
    structure_train, structure_test, y_train, y_test = split_dataset(dataset)

    # TODO: kf = KFold(n_splits=10)
    # TODO: feature_generator = RotationInvariant(structure_train, model_params)

    return model_params
