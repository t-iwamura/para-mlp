import numpy as np

from para_mlp.model import make_content_of_lammps_file


def test_make_content_of_lammps_file(trained_model, seko_lammps_file_lines):
    generated_lines = make_content_of_lammps_file(trained_model).split("\n")

    # Comparison of top blocks
    assert generated_lines[:11] == seko_lammps_file_lines[:11]
    # Comparison of bottom blocks
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


def test_load_model(trained_model, loaded_model, divided_dataset):
    np.testing.assert_allclose(
        trained_model.predict(divided_dataset["test"]["structures"]),
        loaded_model.predict(divided_dataset["test"]["structures"]),
        atol=1e-09,
    )
