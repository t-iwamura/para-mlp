import numpy as np

from para_mlp.model import make_content_of_lammps_file


def test_make_content_of_lammps_file(train_output, seko_lammps_file_lines):
    obtained_model, _ = train_output
    generated_lines = make_content_of_lammps_file(obtained_model).split("\n")

    # Comparison of top parts
    assert generated_lines[:11] == seko_lammps_file_lines[:11]
    # Comparison of bottom parts
    assert generated_lines[-14:] == seko_lammps_file_lines[-14:]

    generated_middle_lines = [
        [float(item) for item in line.split()]
        for line in generated_lines[11:-14]
        if "#" not in line
    ]
    seko_middle_lines = [
        [float(item) for item in line.split()]
        for line in seko_lammps_file_lines[11:-14]
        if "#" not in line
    ]

    assert generated_middle_lines == seko_middle_lines


def test_load_model(train_output, loaded_model_object, divided_dataset):
    obtained_model, obtained_model_params = train_output

    assert obtained_model_params == loaded_model_object["model_params"]

    np.testing.assert_allclose(
        obtained_model.predict(divided_dataset["test"]["structures"]),
        loaded_model_object["model"].predict(divided_dataset["test"]["structures"]),
        atol=1e-09,
    )
