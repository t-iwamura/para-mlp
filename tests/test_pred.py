import numpy as np


def test_load_model(train_output, loaded_model_object, divided_dataset):
    obtained_model, obtained_model_params = train_output

    assert obtained_model_params == loaded_model_object["model_params"]

    np.testing.assert_allclose(
        obtained_model.predict(divided_dataset["test"]["structures"]),
        loaded_model_object["model"].predict(divided_dataset["test"]["structures"]),
        atol=1e-09,
    )
