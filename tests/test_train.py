import numpy as np
import pytest

from para_mlp.train import make_param_grid
from para_mlp.utils import rmse


def test_make_param_grid(test_config):
    param_grid = make_param_grid(test_config)
    expected_param_grid = {
        "cutoff_radius": (6.0, 8.0),
        "gaussian_params2_num": (10),
        "alpha": (1e-2,),
    }
    assert param_grid["cutoff_radius"] == expected_param_grid["cutoff_radius"]
    np.testing.assert_array_equal(
        param_grid["gaussian_params2_num"], expected_param_grid["gaussian_params2_num"]
    )
    assert param_grid["alpha"] == expected_param_grid["alpha"]


def test_train_and_eval(train_output, divided_dataset, n_atoms_in_structure):
    obtained_model, _ = train_output
    test_structures = divided_dataset["test"]["structures"]
    y_predict = obtained_model.predict(test_structures)

    energy_id_end = len(test_structures)
    rmse_energy = (
        rmse(
            y_predict[:energy_id_end] / n_atoms_in_structure,
            divided_dataset["test"]["target"][:energy_id_end] / n_atoms_in_structure,
        )
        * 1e3
    )
    rmse_force = rmse(
        y_predict[energy_id_end:], divided_dataset["test"]["target"][energy_id_end:]
    )

    assert rmse_energy == pytest.approx(8.896594145743832, rel=1e-8)
    assert rmse_force == pytest.approx(0.219520783894536, rel=1e-8)
