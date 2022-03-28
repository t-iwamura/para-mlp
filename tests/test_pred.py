import numpy as np


def test_train_and_eval(train_output, loaded_model_object, feature_of_test_dataset):
    obtained_model, obtained_model_params = train_output

    assert obtained_model_params == loaded_model_object["model_params"]
    np.testing.assert_allclose(
        obtained_model.predict(feature_of_test_dataset),
        loaded_model_object["model"].predict(feature_of_test_dataset),
        rtol=1e-09,
    )
