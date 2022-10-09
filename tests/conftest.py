import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from para_mlp.config import Config
from para_mlp.data_structure import ModelParams
from para_mlp.model import load_model
from para_mlp.preprocess import (
    create_dataset,
    load_ids_for_test_and_kfold,
    make_vasprun_tempfile,
    read_vasprun_tempfile,
    split_dataset,
)
from para_mlp.train import train_and_eval

sys.path.insert(0, "/home/iwamura/usr/local/mlptools/mlp_build_tools/")  # noqa: E402
from common.fileio import InputParams  # noqa: E402
from mlpgen.myIO import ReadFeatureParams  # noqa: E402

tests_dir_path = Path(__file__).resolve().parent
INPUTS_DIR_PATH = tests_dir_path / "data" / "inputs"
OUTPUTS_DIR_PATH = tests_dir_path / "data" / "outputs"
PROCESSING_DIR_PATH = tests_dir_path / "data" / "processing"


@pytest.fixture()
def test_config():
    common_config_dict = {
        "data_dir": "/".join([tests_dir_path.as_posix(), "data"]),
        "targets_json": "/".join(
            [tests_dir_path.as_posix(), "configs", "targets.json"]
        ),
        "cutoff_radius_min": 6.0,
        "cutoff_radius_max": 8.0,
        "gaussian_params2_num_max": 10,
        "gtinv_lmax": (3,),
        "use_spin": False,
        "use_force": True,
        "shuffle": False,
        "alpha": (1e-2,),
        "n_splits": 5,
        "metric": "energy",
        "n_jobs": -1,
    }
    config = Config.from_dict(common_config_dict)

    test_config_dict = {}
    config.composite_num = 1
    test_config_dict["one_specie"] = copy.deepcopy(config)
    config.composite_num = 2
    test_config_dict["two_specie"] = copy.deepcopy(config)

    return test_config_dict


@pytest.fixture()
def test_pair_config():
    config_dict = {
        "composite_num": 2,
        "data_dir": "/".join([str(tests_dir_path), "data"]),
        "targets_json": "/".join([str(tests_dir_path), "configs", "targets.json"]),
        "cutoff_radius_min": 6.0,
        "cutoff_radius_max": 8.0,
        "gaussian_params2_num_max": 10,
        "feature_type": "pair",
        "gtinv_lmax": (0,),
        "use_spin": False,
        "use_force": True,
        "shuffle": False,
        "alpha": (1e-2,),
        "n_splits": 5,
        "n_jobs": -1,
    }
    config = Config.from_dict(config_dict)

    return config


@pytest.fixture()
def model_params_multiconfig(test_config):
    config = test_config["one_specie"]
    common_model_params = {
        "use_force": config.use_force,
        "use_stress": False,
        "polynomial_model": 1,
        "polynomial_max_order": 1,
        "cutoff_radius": 6.0,
        "gaussian_params1": (1.0, 1.0, 1),
        "gaussian_params2": (0.0, 5.0, 10),
        "gtinv_order": 2,
        "gtinv_lmax": config.gtinv_lmax,
        "lmax": config.gtinv_lmax[0],
        "alpha": config.alpha[0],
    }
    model_params = ModelParams.from_dict(common_model_params)

    model_params_dict = {}
    for i, config_key in enumerate(test_config.keys()):
        model_params.composite_num = i + 1
        model_params_dict[config_key] = copy.deepcopy(model_params)

    return model_params_dict


@pytest.fixture()
def model_params_pair(test_pair_config):
    common_model_params = {
        "composite_num": test_pair_config.composite_num,
        "use_force": test_pair_config.use_force,
        "use_stress": False,
        "polynomial_model": 1,
        "polynomial_max_order": 1,
        "cutoff_radius": 6.0,
        "gaussian_params1": (1.0, 1.0, 1),
        "gaussian_params2": (0.0, 5.0, 10),
        "feature_type": test_pair_config.feature_type,
        "gtinv_order": 2,
        "gtinv_lmax": test_pair_config.gtinv_lmax,
        "lmax": test_pair_config.gtinv_lmax[0],
        "alpha": test_pair_config.alpha[0],
    }
    model_params = ModelParams.from_dict(common_model_params)

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
def inputs_dir_path():
    return INPUTS_DIR_PATH


@pytest.fixture()
def outputs_dir_path():
    return OUTPUTS_DIR_PATH


@pytest.fixture()
def vasprun_tempfile_multiconfig(test_config):
    tempfiles = {}
    for config_key in test_config.keys():
        config = test_config[config_key]
        tempfile = make_vasprun_tempfile(
            data_dir=config.data_dir,
            targets_json=config.targets_json,
            composite_num=config.composite_num,
        )
        tempfiles[config_key] = tempfile

    return tempfiles


@pytest.fixture()
def seko_vasprun_outputs_multiconfig(vasprun_tempfile_multiconfig, test_config):
    vasprun_outputs_dict = {}
    for config_key in test_config.keys():
        config = test_config[config_key]
        vasprun_tempfile = vasprun_tempfile_multiconfig[config_key]

        energy, force, seko_structures = read_vasprun_tempfile(
            vasprun_tempfile, composite_num=config.composite_num
        )
        vasprun_outputs_dict[config_key] = copy.deepcopy(
            (energy, force, seko_structures)
        )

    return vasprun_outputs_dict


@pytest.fixture()
def seko_structures_multiconfig(seko_vasprun_outputs_multiconfig):
    seko_structures_dict = {}
    for config_key in seko_vasprun_outputs_multiconfig.keys():
        seko_vasprun_outputs = seko_vasprun_outputs_multiconfig[config_key]
        seko_structures_dict[config_key] = copy.deepcopy(seko_vasprun_outputs[-1])
    return seko_structures_dict


@pytest.fixture()
def dataset_multiconfig(test_config):
    dataset_dict = {}
    for config_key in test_config.keys():
        config = test_config[config_key]
        dataset_dict[config_key] = create_dataset(
            config.data_dir,
            config.targets_json,
            use_force=config.use_force,
            n_jobs=-1,
        )
    return dataset_dict


@pytest.fixture()
def pymatgen_structures_multiconfig(dataset_multiconfig):
    pymatgen_structures_dict = {}
    for config_key in dataset_multiconfig.keys():
        dataset = dataset_multiconfig[config_key]
        pymatgen_structures_dict[config_key] = dataset["structures"]
    return pymatgen_structures_dict


@pytest.fixture()
def n_atoms_in_structure(pymatgen_structures_multiconfig):
    pymatgen_structures = pymatgen_structures_multiconfig["one_specie"]
    return len(pymatgen_structures[0].sites)


@pytest.fixture()
def loaded_ids_for_test_and_kfold(test_config):
    structure_id, yids_for_kfold, yids_for_test = load_ids_for_test_and_kfold(
        str(PROCESSING_DIR_PATH), test_config["one_specie"].use_force
    )

    loaded_ids_dict = {}
    loaded_ids_dict["structure_id"] = structure_id
    loaded_ids_dict["yids_for_kfold"] = yids_for_kfold
    loaded_ids_dict["yids_for_test"] = yids_for_test

    return loaded_ids_dict


@pytest.fixture()
def divided_dataset_ids(dataset_multiconfig, test_config):
    dataset = dataset_multiconfig["one_specie"]
    config = test_config["one_specie"]

    structure_id, yids_for_kfold, yids_for_test = split_dataset(
        dataset, config.use_force, shuffle=False
    )

    return structure_id, yids_for_kfold, yids_for_test


@pytest.fixture()
def divided_dataset_multiconfig(dataset_multiconfig, divided_dataset_ids):
    divided_dataset_dict = {}
    for config_key in dataset_multiconfig.keys():
        dataset = dataset_multiconfig[config_key]

        structure_id, yids_for_kfold, yids_for_test = divided_dataset_ids
        kfold_dataset = {
            "structures": [dataset["structures"][sid] for sid in structure_id["kfold"]],
            "target": dataset["target"][yids_for_kfold["target"]],
        }
        test_dataset = {
            "structures": [dataset["structures"][sid] for sid in structure_id["test"]],
            "target": dataset["target"][yids_for_test["target"]],
        }

        divided_dataset = {"kfold": kfold_dataset, "test": test_dataset}
        divided_dataset_dict[config_key] = copy.deepcopy(divided_dataset)

    return divided_dataset_dict


@pytest.fixture()
def kfold_feature_by_seko_method_multiconfig(test_config):
    """Feature matrix outputed by get_xy() in regression.py"""
    kfold_feature_dict = {}
    for config_key in test_config.keys():
        kfold_feature_path = PROCESSING_DIR_PATH / config_key / "kfold_feature.npy"
        kfold_feature = np.load(kfold_feature_path)
        kfold_feature_dict[config_key] = copy.deepcopy(kfold_feature)

    return kfold_feature_dict


@pytest.fixture()
def pair_feature_by_seko_method():
    pair_feature_path = PROCESSING_DIR_PATH / "pair" / "kfold_feature.npy"
    pair_feature = np.load(pair_feature_path)

    return pair_feature


@pytest.fixture()
def spin_energy_feature_832():
    spin_feature_path = PROCESSING_DIR_PATH / "00832" / "spin_energy_feature.json"
    with spin_feature_path.open("r") as f:
        spin_feature = json.load(f)

    return spin_feature


@pytest.fixture()
def spin_force_feature_832():
    spin_feature_path = PROCESSING_DIR_PATH / "00832" / "spin_force_feature.json"
    with spin_feature_path.open("r") as f:
        spin_feature = json.load(f)

    return spin_feature


@pytest.fixture()
def trained_model_multiconfig(test_config, divided_dataset_multiconfig):
    obtained_model_dict = {}
    for config_key in test_config.keys():
        config = test_config[config_key]
        divided_dataset = divided_dataset_multiconfig[config_key]

        obtained_model = train_and_eval(
            config, divided_dataset["kfold"], divided_dataset["test"]
        )
        obtained_model_dict[config_key] = copy.deepcopy(obtained_model)

    return obtained_model_dict


@pytest.fixture()
def loaded_model_multiconfig(test_config):
    loaded_model_dict = {}
    for config_key in test_config.keys():
        model_dir_path = OUTPUTS_DIR_PATH / config_key
        loaded_model = load_model(model_dir_path.as_posix())
        loaded_model_dict[config_key] = copy.deepcopy(loaded_model)

    return loaded_model_dict


@pytest.fixture()
def seko_model_params_multiconfig(test_config):
    seko_model_params_dict = {}
    for config_key in test_config.keys():
        seko_input_filepath = INPUTS_DIR_PATH / "seko_input" / config_key / "train.in"
        input_params = InputParams(seko_input_filepath.as_posix())
        seko_model_params = ReadFeatureParams(input_params).get_params()
        seko_model_params_dict[config_key] = copy.deepcopy(seko_model_params)
    return seko_model_params_dict


@pytest.fixture()
def seko_struct_params_multiconfig(seko_structures_multiconfig, test_config):
    seko_struct_params_dict = {}
    for config_key in test_config.keys():
        seko_structures = seko_structures_multiconfig[config_key]
        struct_params = {}

        struct_params["axis_array"] = [struct.get_axis() for struct in seko_structures]
        struct_params["positions_c_array"] = [
            struct.get_positions_cartesian() for struct in seko_structures
        ]
        struct_params["types_array"] = [
            struct.get_types() for struct in seko_structures
        ]
        struct_params["n_atoms_all"] = [
            sum(struct.get_n_atoms()) for struct in seko_structures
        ]
        struct_params["n_st_dataset"] = [len(seko_structures)]

        seko_struct_params_dict[config_key] = copy.deepcopy(struct_params)

    return seko_struct_params_dict


@pytest.fixture()
def seko_lammps_file_lines_multiconfig(test_config):
    seko_lammps_file_lines_dict = {}
    for config_key in test_config.keys():
        lammps_file_path = OUTPUTS_DIR_PATH / config_key / "mlp.lammps"
        with lammps_file_path.open("r") as f:
            content = f.read()
        seko_lammps_file_lines_dict[config_key] = content.split("\n")
    return seko_lammps_file_lines_dict
