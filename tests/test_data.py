import os

import numpy as np
from pymatgen.core.structure import Structure

from para_mlp.featurize import RotationInvariant

inputs_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/inputs/data/"

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


def test_create_dataset(pymatgen_structures):
    target_structures = pymatgen_structures
    ref_structures = [Structure.from_file(inputs_dir + si + "/CONTCAR") for si in structure_ids]
    np.testing.assert_allclose(
        [struct.lattice.matrix for struct in target_structures],
        [struct.lattice.matrix for struct in ref_structures],
        rtol=2e-6,
    )


def test_rotation_invariant(pymatgen_structures, args_for_term):
    feature = RotationInvariant(pymatgen_structures)
    np.testing.assert_array_equal(feature.axis_array, args_for_term["axis_array"])
    np.testing.assert_array_equal(
        feature.positions_c_array,
        args_for_term["positions_c_array"],
    )
    np.testing.assert_array_equal(
        feature.types_array,
        args_for_term["types_array"],
    )
    np.testing.assert_array_equal(
        feature.n_st_dataset,
        args_for_term["n_st_dataset"],
    )
    np.testing.assert_array_equal(
        feature.n_atoms_all,
        args_for_term["n_atoms_all"],
    )
