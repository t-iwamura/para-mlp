import os

import numpy as np
from pymatgen.core.structure import Structure

from para_mlp.featurize import RotationInvariant

inputs_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/inputs/data/"


def test_create_dataset(pymatgen_structures, structure_ids):
    target_structures = pymatgen_structures
    ref_structures = [Structure.from_file(inputs_dir + si + "/CONTCAR") for si in structure_ids]
    np.testing.assert_allclose(
        [struct.lattice.matrix for struct in target_structures],
        [struct.lattice.matrix for struct in ref_structures],
        rtol=2e-6,
    )


def test_structures_for_invariant(pymatgen_structures, args_for_term):
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


def test_model_params_for_invariant(model_params, seko_model_params):
    np.testing.assert_equal(
        model_params["composite_num"],
        seko_model_params.n_type,
    )
    np.testing.assert_equal(
        model_params["radial_params"],
        seko_model_params.model_e.des_params,
    )
    np.testing.assert_equal(
        model_params["cutoff_radius"],
        seko_model_params.model_e.cutoff,
    )
    np.testing.assert_equal(
        model_params["radial_func"],
        seko_model_params.model_e.pair_type,
    )
    np.testing.assert_equal(
        model_params["feature_type"],
        seko_model_params.model_e.des_type,
    )
    np.testing.assert_equal(
        model_params["polynomial_max_order"],
        seko_model_params.model_e.maxp,
    )
    np.testing.assert_equal(
        model_params["lmax"],
        seko_model_params.model_e.maxl,
    )
    np.testing.assert_equal(
        model_params["lm_seq"],
        seko_model_params.model_e.lm_seq,
    )
    np.testing.assert_equal(
        model_params["l_comb"],
        seko_model_params.model_e.l_comb,
    )
    np.testing.assert_equal(
        model_params["lm_coeffs"],
        seko_model_params.model_e.lm_coeffs,
    )
