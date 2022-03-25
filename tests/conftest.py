import copy
import os

import pytest
from mlp_build_tools.common.fileio import InputParams
from mlp_build_tools.mlpgen.myIO import ReadFeatureParams, ReadVaspruns

from para_mlp.data_structure import make_model_params
from para_mlp.preprocess import create_dataset, make_vasprun_tempfile

inputs_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/inputs/"


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def structures(structure_ids):
    vasprun_tempfile = make_vasprun_tempfile(structure_ids)

    energy, force, stress, structures, volume = ReadVaspruns(vasprun_tempfile).get_data()

    return structures


@pytest.fixture(scope="session")
def pymatgen_structures(structure_ids):
    return create_dataset(structure_ids)["structures"]


@pytest.fixture(scope="session")
def model_params():
    model_params = {}
    model_params["use_force"] = False
    model_params["use_stress"] = False
    model_params["composite_num"] = 1
    model_params["polynomial_model"] = 1
    model_params["polynomial_max_order"] = 2
    model_params["feature_type"] = "gtinv"
    model_params["cutoff_radius"] = 6.0
    model_params["radial_func"] = "gaussian"
    model_params["atomic_energy"] = -3.37689

    hyper_params = {}
    hyper_params["gaussian_params1"] = (1.0, 1.0, 1)
    hyper_params["gaussian_params2"] = (1.0, 5.0, 10)
    hyper_params["gtinv_order"] = 2
    hyper_params["gtinv_lmax"] = [3]
    hyper_params["gtinv_sym"] = [False]
    model_params["lmax"] = copy.copy(hyper_params["gtinv_lmax"])[0]

    model_params.update(make_model_params(hyper_params))

    return model_params


@pytest.fixture(scope="session")
def seko_model_params():
    input_params = InputParams(inputs_dir + "train.in")
    seko_model_params = ReadFeatureParams(input_params).get_params()

    return seko_model_params


@pytest.fixture(scope="session")
def seko_struct_params(structures):
    struct_params = {}
    struct_params["axis_array"] = [struct.get_axis() for struct in structures]
    struct_params["positions_c_array"] = [struct.get_positions_cartesian() for struct in structures]
    struct_params["types_array"] = [struct.get_types() for struct in structures]
    struct_params["n_atoms_all"] = [sum(struct.get_n_atoms()) for struct in structures]
    struct_params["n_st_dataset"] = [len(structures)]

    return struct_params
