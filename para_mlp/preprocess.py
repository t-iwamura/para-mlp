import os
import sys
from itertools import product
from tempfile import NamedTemporaryFile

import numpy as np
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../cpp/lib")


def make_model_params(hyper_params: dict, model_params: dict):
    import mlpcpp

    rotation_invariant = mlpcpp.Readgtinv(hyper_params["gtinv_order"], hyper_params["gtinv_lmax"], hyper_params["gtinv_sym"], 1)
    model_params["lm_seq"] = rotation_invariant.get_lm_seq()
    model_params["l_comb"] = rotation_invariant.get_l_comb()
    model_params["lm_coeffs"] = rotation_invariant.get_lm_coeffs()

    radial_params = hyper_params["gaussian_params1"]
    radial_params1 = np.linspace(radial_params[0], radial_params[1], radial_params[2])
    radial_params = hyper_params["gaussian_params2"]
    radial_params2 = np.linspace(radial_params[0], radial_params[1], radial_params[2])
    model_params["radial_params"] = list(product(radial_params1, radial_params2))

    return model_params


def create_dataset(structure_ids: tuple = None):
    if structure_ids is None:
        structure_ids = (str(i + 1).zfill(5) for i in range(5000))

    inputs_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/inputs/data/"
    f = NamedTemporaryFile(mode="w", delete=False)
    for si in structure_ids:
        print(inputs_dir + si + "/vasprun.xml_1_type", file=f)
    f.close()
    energy, force, stress, seko_structures, volume = ReadVaspruns(f.name).get_data()

    structures = [
        Structure(struct.get_axis().transpose(), struct.get_elements(), struct.get_positions().transpose())
        for struct in seko_structures
    ]

    dataset = {}
    dataset["energy"] = energy
    dataset["structures"] = structures

    return dataset


def split_dataset(dataset: dict = None, test_size: float = 0.1):
    structure_train, structure_test, y_train, y_test = train_test_split(
        dataset["structures"], dataset["energies"], test_size=test_size, shuffle=True
    )

    return (structure_train, structure_test, y_train, y_test)
