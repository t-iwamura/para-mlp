import numpy as np

from para_mlp.train import make_param_grid


def test_make_param_grid(test_config):
    param_grid = make_param_grid(test_config)
    expected_param_grid = {
        "cutoff_radius": (6.0, 8.0),
        "gaussian_params2_num": (5, 10),
        "alpha": (1e-2,),
    }
    assert param_grid["cutoff_radius"] == expected_param_grid["cutoff_radius"]
    np.testing.assert_array_equal(
        param_grid["gaussian_params2_num"], expected_param_grid["gaussian_params2_num"]
    )
    assert param_grid["alpha"] == expected_param_grid["alpha"]
