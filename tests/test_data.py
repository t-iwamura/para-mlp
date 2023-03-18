from pathlib import Path

import numpy as np
from pymatgen.core.structure import Structure

from para_mlp.featurize import RotationInvariant

inputs_dir_path = Path(__file__).resolve().parent / "data" / "inputs" / "sqs" / "data"
inputs_dir = str(inputs_dir_path)


def test_create_dataset(pymatgen_structures_multiconfig, structure_ids):
    for config_key in pymatgen_structures_multiconfig.keys():
        target_structures = pymatgen_structures_multiconfig[config_key]
        assert len(target_structures) == len(structure_ids)

        ref_structures = [
            Structure.from_file("/".join([inputs_dir, si, "POSCAR"]))
            for si in structure_ids
        ]
        np.testing.assert_allclose(
            [struct.lattice.matrix for struct in target_structures],
            [struct.lattice.matrix for struct in ref_structures],
            atol=1e-6,
        )


def test_struct_params_for_invariant(
    pymatgen_structures_multiconfig,
    seko_struct_params_multiconfig,
    model_params_multiconfig,
):
    for config_key in pymatgen_structures_multiconfig.keys():
        pymatgen_structures = pymatgen_structures_multiconfig[config_key]
        seko_struct_params = seko_struct_params_multiconfig[config_key]
        model_params = model_params_multiconfig[config_key]

        ri = RotationInvariant(model_params)
        (
            axis_array,
            positions_c_array,
            types_array,
            n_st_dataset,
            n_atoms_all,
        ) = ri.make_struct_params(pymatgen_structures)
        np.testing.assert_array_equal(axis_array, seko_struct_params["axis_array"])
        np.testing.assert_allclose(
            positions_c_array,
            seko_struct_params["positions_c_array"],
            rtol=1e-10,
        )
        np.testing.assert_array_equal(
            types_array,
            seko_struct_params["types_array"],
        )
        np.testing.assert_array_equal(
            n_st_dataset,
            seko_struct_params["n_st_dataset"],
        )
        np.testing.assert_array_equal(
            n_atoms_all,
            seko_struct_params["n_atoms_all"],
        )


def test_model_params_for_invariant(
    model_params_multiconfig, seko_model_params_multiconfig
):
    for config_key in model_params_multiconfig.keys():
        model_params = model_params_multiconfig[config_key]
        feature_params = model_params.make_feature_params()
        seko_model_params = seko_model_params_multiconfig[config_key]

        np.testing.assert_equal(
            model_params.composite_num,
            seko_model_params.n_type,
        )
        np.testing.assert_equal(
            feature_params["radial_params"],
            seko_model_params.model_e.des_params,
        )
        np.testing.assert_equal(
            model_params.cutoff_radius,
            seko_model_params.model_e.cutoff,
        )
        np.testing.assert_equal(
            model_params.radial_func,
            seko_model_params.model_e.pair_type,
        )
        np.testing.assert_equal(
            model_params.feature_type,
            seko_model_params.model_e.des_type,
        )
        np.testing.assert_equal(
            model_params.polynomial_max_order,
            seko_model_params.model_e.maxp,
        )
        np.testing.assert_equal(
            model_params.lmax,
            seko_model_params.model_e.maxl,
        )
        np.testing.assert_equal(
            feature_params["lm_seq"],
            seko_model_params.model_e.lm_seq,
        )
        np.testing.assert_equal(
            feature_params["l_comb"],
            seko_model_params.model_e.l_comb,
        )
        np.testing.assert_equal(
            feature_params["lm_coeffs"],
            seko_model_params.model_e.lm_coeffs,
        )
