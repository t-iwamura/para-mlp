from pathlib import Path

import numpy as np
from pymatgen.core.structure import Structure

from para_mlp.featurize import RotationInvariant

inputs_dir_path = Path(__file__).resolve().parent / "data" / "inputs" / "data"
inputs_dir = inputs_dir_path.as_posix()


def test_create_dataset(pymatgen_structures, structure_ids):
    target_structures = pymatgen_structures
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
    pymatgen_structures, seko_struct_params, model_params
):
    ri = RotationInvariant(model_params)
    ri.set_struct_params(pymatgen_structures)
    np.testing.assert_array_equal(ri.axis_array, seko_struct_params["axis_array"])
    np.testing.assert_allclose(
        ri.positions_c_array,
        seko_struct_params["positions_c_array"],
        rtol=1e-10,
    )
    np.testing.assert_array_equal(
        ri.types_array,
        seko_struct_params["types_array"],
    )
    np.testing.assert_array_equal(
        ri.n_st_dataset,
        seko_struct_params["n_st_dataset"],
    )
    np.testing.assert_array_equal(
        ri.n_atoms_all,
        seko_struct_params["n_atoms_all"],
    )


def test_model_params_for_invariant(model_params, seko_model_params):
    np.testing.assert_equal(
        model_params.composite_num,
        seko_model_params.n_type,
    )
    np.testing.assert_equal(
        model_params.radial_params,
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
        model_params.lm_seq,
        seko_model_params.model_e.lm_seq,
    )
    np.testing.assert_equal(
        model_params.l_comb,
        seko_model_params.model_e.l_comb,
    )
    np.testing.assert_equal(
        model_params.lm_coeffs,
        seko_model_params.model_e.lm_coeffs,
    )
