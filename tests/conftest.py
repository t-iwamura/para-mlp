import json
from pathlib import Path

import numpy as np
import pytest
from mlp_build_tools.common.fileio import InputParams
from mlp_build_tools.mlpgen.myIO import ReadFeatureParams

from para_mlp.config import Config
from para_mlp.data_structure import ModelParams
from para_mlp.preprocess import (
    create_dataset,
    make_vasprun_tempfile,
    read_vasprun_tempfile,
    split_dataset,
)
from para_mlp.train import load_model, train_and_eval

tests_dir_path = Path(__file__).resolve().parent
INPUTS_DIR_PATH = tests_dir_path / "data" / "inputs" / "seko_input"
OUTPUTS_DIR_PATH = tests_dir_path / "data" / "outputs"
PROCESSING_DIR_PATH = tests_dir_path / "data" / "processing"


@pytest.fixture()
def test_config():
    config_dict = {
        "data_dir": "/".join([tests_dir_path.as_posix(), "data"]),
        "targets_json": "/".join(
            [tests_dir_path.as_posix(), "configs", "targets.json"]
        ),
        "cutoff_radius_min": 6.0,
        "cutoff_radius_max": 8.0,
        "gaussian_params2_num_max": 10,
        "gtinv_lmax": (3,),
        "use_force": True,
        "shuffle": False,
        "alpha": (1e-2,),
        "n_jobs": -1,
    }
    config = Config.from_dict(config_dict)

    return config


@pytest.fixture()
def model_params(test_config):
    model_params_dict = {
        "use_force": test_config.use_force,
        "use_stress": False,
        "polynomial_model": 1,
        "polynomial_max_order": 1,
        "cutoff_radius": 6.0,
        "gaussian_params1": (1.0, 1.0, 1),
        "gaussian_params2": (0.0, 6.0, 5),
        "gtinv_order": 2,
        "gtinv_lmax": test_config.gtinv_lmax,
        "alpha": test_config.alpha[0],
    }
    model_params = ModelParams.from_dict(model_params_dict)
    model_params.make_feature_params()

    return model_params


# same as structure ids in tests/configs/targets.json
@pytest.fixture()
def structure_ids():
    structure_ids = (
        "04075",
        "03299",
        "01747",
        "00080",
        "00903",
        "04960",
        "03461",
        "00533",
        "03715",
        "00262",
        "04624",
        "00795",
        "04061",
        "03834",
        "03263",
        "04214",
        "00644",
        "04212",
        "02678",
        "04611",
        "02856",
        "03895",
        "00615",
        "02612",
        "00681",
        "03194",
        "03445",
        "00924",
        "02357",
        "00275",
        "04218",
        "01795",
        "01823",
        "02446",
        "00979",
        "03285",
        "04826",
        "01513",
        "01842",
        "00723",
        "04230",
        "04511",
        "00018",
        "00994",
        "01502",
        "00657",
        "03846",
        "01516",
        "02200",
        "00485",
        "04412",
        "04157",
        "03765",
        "04371",
        "00863",
        "04482",
        "01043",
        "04220",
        "02954",
        "03809",
        "00702",
        "01806",
        "04451",
        "02860",
        "04024",
        "00543",
        "04244",
        "04773",
        "02006",
        "04398",
        "02792",
        "02049",
        "02246",
        "01238",
        "03635",
        "02953",
        "00749",
        "01574",
        "00146",
        "04000",
        "00931",
        "02647",
        "04965",
        "02229",
        "02171",
        "00258",
        "02733",
        "04786",
        "02576",
        "04869",
        "02952",
        "01155",
        "04146",
        "02788",
        "02483",
        "03805",
        "03037",
        "01334",
        "02020",
        "00832",
    )

    return structure_ids


@pytest.fixture()
def vasprun_tempfile(test_config):
    tempfile = make_vasprun_tempfile(
        data_dir=test_config.data_dir, targets_json=test_config.targets_json
    )

    return tempfile


@pytest.fixture()
def seko_vasprun_outputs(vasprun_tempfile):
    energy, force, seko_structures = read_vasprun_tempfile(vasprun_tempfile)

    return energy, force, seko_structures


@pytest.fixture()
def seko_structures(seko_vasprun_outputs):
    return seko_vasprun_outputs[-1]


@pytest.fixture()
def dataset(test_config):
    return create_dataset(
        test_config.data_dir,
        test_config.targets_json,
        use_force=test_config.use_force,
        n_jobs=-1,
    )


@pytest.fixture()
def pymatgen_structures(dataset):
    return dataset["structures"]


@pytest.fixture()
def divided_dataset(dataset):
    yid, structure_id = split_dataset(dataset, shuffle=False)
    test_dataset = {
        "structures": [dataset["structures"][sid] for sid in structure_id["test"]],
        "target": dataset["target"][yid["test"]],
    }
    kfold_dataset = {
        "structures": [dataset["structures"][sid] for sid in structure_id["kfold"]],
        "target": dataset["target"][yid["kfold"]],
    }

    divided_dataset = {"kfold": kfold_dataset, "test": test_dataset}

    return divided_dataset


@pytest.fixture()
def kfold_feature_by_seko_method():
    # Feature matrix outputed by get_xy() in regression.py
    kfold_feature_path = PROCESSING_DIR_PATH / "kfold_feature.npy"
    kfold_feature = np.load(kfold_feature_path)

    return kfold_feature


@pytest.fixture()
def spin_feature_832():
    spin_feature_path = PROCESSING_DIR_PATH / "00832" / "spin_feature.json"
    with spin_feature_path.open("r") as f:
        spin_feature = json.load(f)

    return spin_feature


@pytest.fixture()
def train_output(test_config, divided_dataset):
    obtained_model, obtained_model_params = train_and_eval(
        test_config, divided_dataset["kfold"], divided_dataset["test"]
    )

    return obtained_model, obtained_model_params


@pytest.fixture()
def loaded_model_object():
    loaded_model, loaded_model_params = load_model(OUTPUTS_DIR_PATH.as_posix())

    model_object = {"model": loaded_model, "model_params": loaded_model_params}

    return model_object


@pytest.fixture()
def seko_model_params():
    seko_input_filepath = INPUTS_DIR_PATH / "train.in"
    input_params = InputParams(seko_input_filepath.as_posix())
    seko_model_params = ReadFeatureParams(input_params).get_params()

    return seko_model_params


@pytest.fixture()
def seko_struct_params(seko_structures):
    struct_params = {}
    struct_params["axis_array"] = [struct.get_axis() for struct in seko_structures]
    struct_params["positions_c_array"] = [
        struct.get_positions_cartesian() for struct in seko_structures
    ]
    struct_params["types_array"] = [struct.get_types() for struct in seko_structures]
    struct_params["n_atoms_all"] = [
        sum(struct.get_n_atoms()) for struct in seko_structures
    ]
    struct_params["n_st_dataset"] = [len(seko_structures)]

    return struct_params
