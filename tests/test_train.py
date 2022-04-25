import pytest

from para_mlp.train import make_param_grid
from para_mlp.utils import rmse


def test_make_param_grid(test_config):
    param_grid = make_param_grid(test_config)
    expected_param_grid = {
        "cutoff_radius": (6.0, 8.0),
        "gaussian_params2_num": (10,),
        "alpha": (1e-2,),
    }
    assert param_grid == expected_param_grid


def test_train_and_eval(trained_model, divided_dataset, n_atoms_in_structure):
    test_structures = divided_dataset["test"]["structures"]
    y_predict = trained_model.predict(test_structures)

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

    assert rmse_energy == pytest.approx(10.254324039723619, rel=1e-7)
    assert rmse_force == pytest.approx(0.21943135439411288, rel=1e-8)
