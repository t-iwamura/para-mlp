import pytest

from para_mlp.train import make_param_grid
from para_mlp.utils import rmse


def test_make_param_grid(test_config):
    config = test_config["one_specie"]

    param_grid = make_param_grid(config)
    expected_param_grid = {
        "cutoff_radius": (6.0, 8.0),
        "gaussian_params2_num": (10,),
        "alpha": (1e-2,),
    }
    assert param_grid == expected_param_grid

    config.cutoff_radius_min = 7.0
    config.cutoff_radius_max = 7.0
    param_grid = make_param_grid(config)
    expected_param_grid["cutoff_radius"] = (7.0,)
    assert param_grid == expected_param_grid


def test_train_and_eval(
    trained_model_multiconfig, divided_dataset_multiconfig, n_atoms_in_structure
):
    seko_rmse_energy = {
        "one_specie": 10.254324039723619,
        "two_specie": 9.496509871532915,
    }
    seko_rmse_force = {
        "one_specie": 0.21943135439411288,
        "two_specie": 0.20596040946178956,
    }

    for config_key in trained_model_multiconfig.keys():
        trained_model = trained_model_multiconfig[config_key]
        divided_dataset = divided_dataset_multiconfig[config_key]

        test_structures = divided_dataset["test"]["structures"]
        n_structure_list = divided_dataset["test"]["n_structure"]
        y_predict = trained_model.predict(test_structures, n_structure_list)

        energy_id_end = len(test_structures)
        rmse_energy = (
            rmse(
                y_predict[:energy_id_end] / n_atoms_in_structure,
                divided_dataset["test"]["target"][:energy_id_end]
                / n_atoms_in_structure,
            )
            * 1e3
        )
        rmse_force = rmse(
            y_predict[energy_id_end:], divided_dataset["test"]["target"][energy_id_end:]
        )

        assert rmse_energy == pytest.approx(seko_rmse_energy[config_key], rel=1e-7)
        assert rmse_force == pytest.approx(seko_rmse_force[config_key], rel=1e-8)
