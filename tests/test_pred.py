import numpy as np


def test_train_and_eval(train_output, loaded_model_object, divided_dataset):
    obtained_model, obtained_model_params = train_output

    assert obtained_model_params == loaded_model_object["model_params"]

    np.testing.assert_allclose(
        obtained_model(divided_dataset["test"]["structures"]),
        loaded_model_object["model"](divided_dataset["test"]["structures"]),
        rtol=1e-09,
    )
