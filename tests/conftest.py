from pathlib import Path

import pytest
from mlp_build_tools.common.fileio import InputParams
from mlp_build_tools.mlpgen.myIO import ReadFeatureParams, ReadVaspruns

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant
from para_mlp.preprocess import create_dataset, make_vasprun_tempfile, split_dataset
from para_mlp.train import load_model, train_and_eval

tests_dir_path = Path(__file__).resolve().parent
inputs_dir_path = tests_dir_path / "data" / "inputs" / "seko_input"
outputs_dir_path = tests_dir_path / "data" / "outputs"

data_dir = "/".join([tests_dir_path.as_posix(), "data"])
targets_json = "/".join([tests_dir_path.as_posix(), "configs", "targets.json"])


# Same as structure ids in tests/configs/targets.json
@pytest.fixture()
def structure_ids():
    structure_ids = (
        "00287",
        "03336",
        "04864",
        "04600",
        "04548",
        "00806",
        "04923",
        "02915",
        "02355",
        "03636",
        "00294",
        "00979",
        "04003",
        "04724",
        "03138",
        "04714",
        "01443",
        "00299",
        "02565",
        "00221",
        "02815",
        "01577",
        "03975",
        "00428",
        "01278",
        "00944",
        "04715",
        "00595",
        "04050",
        "02256",
        "03725",
        "02363",
        "00028",
        "02190",
        "02807",
        "01030",
        "04941",
        "03616",
        "03764",
        "02430",
        "03366",
        "04241",
        "04232",
        "02588",
        "02507",
        "01563",
        "01816",
        "04436",
        "04655",
        "01838",
    )

    return structure_ids


@pytest.fixture()
def structures():
    vasprun_tempfile = make_vasprun_tempfile(
        data_dir=data_dir, targets_json=targets_json
    )

    energy, force, stress, structures, volume = ReadVaspruns(
        vasprun_tempfile
    ).get_data()

    return structures


@pytest.fixture()
def dataset():
    return create_dataset(data_dir=data_dir, targets_json=targets_json)


@pytest.fixture()
def pymatgen_structures(dataset):
    return dataset["structures"]


@pytest.fixture()
def divided_dataset(dataset):
    kfold_dataset, test_dataset = split_dataset(dataset, shuffle=False)

    divided_dataset = {"kfold": kfold_dataset, "test": test_dataset}

    return divided_dataset


@pytest.fixture()
def train_output(model_params, divided_dataset):
    obtained_model, obtained_model_params = train_and_eval(
        model_params, divided_dataset["kfold"], divided_dataset["test"]
    )

    return obtained_model, obtained_model_params


@pytest.fixture()
def loaded_model_object():
    loaded_model, loaded_model_params = load_model(outputs_dir_path.as_posix())

    model_object = {"model": loaded_model, "model_params": loaded_model_params}

    return model_object


@pytest.fixture()
def feature_of_test_dataset(loaded_model_object, divided_dataset):
    ri = RotationInvariant(loaded_model_object["model_params"])

    return ri(divided_dataset["test"]["structures"])


@pytest.fixture()
def model_params():
    model_params = ModelParams()
    model_params.make_feature_params()

    return model_params


@pytest.fixture()
def seko_model_params():
    seko_input_filepath = inputs_dir_path / "train.in"
    input_params = InputParams(seko_input_filepath.as_posix())
    seko_model_params = ReadFeatureParams(input_params).get_params()

    return seko_model_params


@pytest.fixture()
def seko_struct_params(structures):
    struct_params = {}
    struct_params["axis_array"] = [struct.get_axis() for struct in structures]
    struct_params["positions_c_array"] = [
        struct.get_positions_cartesian() for struct in structures
    ]
    struct_params["types_array"] = [struct.get_types() for struct in structures]
    struct_params["n_atoms_all"] = [sum(struct.get_n_atoms()) for struct in structures]
    struct_params["n_st_dataset"] = [len(structures)]

    return struct_params
