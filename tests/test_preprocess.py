import numpy as np
import pytest

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant, SpinFeaturizer
from para_mlp.preprocess import make_force_id, make_high_energy_yids, merge_sub_dataset


@pytest.mark.parametrize(
    ("sid", "atom_id", "force_comp", "expected"),
    [("00287", 17, 1, 27508), ("00001", 0, 0, 0), ("5000", 31, 2, 479999)],
)
def test_make_force_id(sid, atom_id, force_comp, expected):
    assert make_force_id(sid, atom_id, force_comp, n_atom=32) == expected


def test_make_high_energy_yids(
    high_energy_sids,
    yids_for_kfold_high_energy,
    expected_high_energy_yids,
):
    high_energy_yids = make_high_energy_yids(
        high_energy_sids=high_energy_sids,
        n_structure=100,
        force_id_unit=12,
        yids_for_kfold=yids_for_kfold_high_energy,
        use_force=True,
    )
    np.testing.assert_equal(high_energy_yids, expected_high_energy_yids)


def test_make_high_energy_struct_dicts():
    pass


def test_load_vasp_outputs(dataset_multiconfig, seko_vasprun_outputs_multiconfig):
    for config_key in dataset_multiconfig:
        dataset = dataset_multiconfig[config_key]
        energy, force, _ = seko_vasprun_outputs_multiconfig[config_key]

        np.testing.assert_allclose(
            dataset["target"],
            np.concatenate((energy, force), axis=0),
            rtol=1e-13,
        )


def test_load_ids_for_test_and_kfold(
    divided_dataset_ids, loaded_ids_for_test_and_kfold
):
    structure_id, yids_for_kfold, yids_for_test = divided_dataset_ids

    assert structure_id == loaded_ids_for_test_and_kfold["structure_id"]
    assert yids_for_kfold == loaded_ids_for_test_and_kfold["yids_for_kfold"]
    assert yids_for_test == loaded_ids_for_test_and_kfold["yids_for_test"]


def test_split_dataset(divided_dataset_multiconfig, dataset_multiconfig):
    for config_key in divided_dataset_multiconfig.keys():
        divided_dataset = divided_dataset_multiconfig[config_key]
        dataset = dataset_multiconfig[config_key]

        n_test_structure = divided_dataset["test"]["target"].shape[0] // 97
        n_kfold_structure = divided_dataset["kfold"]["target"].shape[0] // 97
        energy = np.concatenate(
            (
                divided_dataset["test"]["target"][:n_test_structure],
                divided_dataset["kfold"]["target"][:n_kfold_structure],
            ),
            axis=0,
        )
        force = np.concatenate(
            (
                divided_dataset["test"]["target"][n_test_structure:],
                divided_dataset["kfold"]["target"][n_kfold_structure:],
            ),
            axis=0,
        )
        np.testing.assert_array_equal(
            np.concatenate(
                (energy, force),
                axis=0,
            ),
            dataset["target"],
        )


def test_merge_sub_dataset(all_dataset, expected_merged_dataset):
    kfold_dataset, test_dataset = merge_sub_dataset(
        all_dataset=all_dataset, data_dir_list=["sqs", "fm"]
    )

    expected_kfold_dataset, expected_test_dataset = expected_merged_dataset
    np.testing.assert_allclose(
        [struct.lattice.matrix for struct in test_dataset["structures"]],
        [struct.lattice.matrix for struct in expected_test_dataset["structures"]],
        atol=1e-6,
    )
    assert test_dataset["types_list"] == expected_test_dataset["types_list"]
    np.testing.assert_array_equal(
        test_dataset["energy"],
        expected_test_dataset["energy"],
    )
    np.testing.assert_array_equal(
        test_dataset["force"],
        expected_test_dataset["force"],
    )
    assert test_dataset["n_structure"] == [10, 1]
    assert test_dataset["n_atom_in_structures"] == [32, 4]

    np.testing.assert_allclose(
        [struct.lattice.matrix for struct in kfold_dataset["structures"]],
        [struct.lattice.matrix for struct in expected_kfold_dataset["structures"]],
        atol=1e-6,
    )
    assert kfold_dataset["types_list"] == expected_kfold_dataset["types_list"]
    np.testing.assert_array_equal(
        kfold_dataset["energy"],
        expected_kfold_dataset["energy"],
    )
    np.testing.assert_array_equal(
        kfold_dataset["force"],
        expected_kfold_dataset["force"],
    )
    assert kfold_dataset["n_structure"] == [90, 9]
    assert kfold_dataset["n_atom_in_structures"] == [32, 4]


@pytest.mark.parametrize(
    (
        "gtinv_lmax",
        "use_gtinv_sym",
        "cutoff_radius",
        "gaussian_params2_num",
        "gaussian_params2",
        "gtinv_order",
        "gtinv_sym",
        "gaussian_params2_flag",
    ),
    [
        ((8, 5, 3, 2), False, 6.0, 5, (0.0, 5.0, 5), 5, (False,) * 4, 1),
        ((6, 2, 2), False, 8.0, 10, (0.0, 7.0, 10), 4, (False,) * 3, 1),
        ((8, 5, 3, 2), False, 6.0, 5, (0.0, 4.8, 5), 5, (False,) * 4, 2),
        ((6, 2, 2), False, 8.0, 10, (0.0, 7.2, 10), 4, (False,) * 3, 2),
        ((6, 2, 2), False, 7.0, 10, (0.0, 6.3, 10), 4, (False,) * 3, 2),
    ],
)
def test_set_api_params(
    gtinv_lmax,
    use_gtinv_sym,
    cutoff_radius,
    gaussian_params2_num,
    gaussian_params2,
    gtinv_order,
    gtinv_sym,
    gaussian_params2_flag,
):
    model_params = ModelParams()
    model_params.gtinv_lmax = gtinv_lmax
    model_params.use_gtinv_sym = use_gtinv_sym
    model_params.cutoff_radius = cutoff_radius
    model_params.gaussian_params2_num = gaussian_params2_num
    model_params.gaussian_params2_flag = gaussian_params2_flag

    model_params.set_api_params()
    assert model_params.gaussian_params2 == gaussian_params2
    assert model_params.gtinv_order == gtinv_order
    assert model_params.gtinv_sym == gtinv_sym


def test_rotation_invariant(
    model_params_multiconfig,
    divided_dataset_multiconfig,
    kfold_feature_by_seko_method_multiconfig,
):
    for config_key in model_params_multiconfig.keys():
        model_params = model_params_multiconfig[config_key]
        divided_dataset = divided_dataset_multiconfig[config_key]
        kfold_feature_by_seko_method = kfold_feature_by_seko_method_multiconfig[
            config_key
        ]

        ri = RotationInvariant(model_params)
        np.testing.assert_allclose(
            ri(
                divided_dataset["kfold"]["structures"],
                divided_dataset["kfold"]["n_structure"],
            ),
            kfold_feature_by_seko_method,
            rtol=1e-8,
        )


def test_pair_feature(
    model_params_pair, divided_dataset_multiconfig, pair_feature_by_seko_method
):
    ri = RotationInvariant(model_params_pair)
    np.testing.assert_allclose(
        ri(
            divided_dataset_multiconfig["two_specie"]["kfold"]["structures"],
            divided_dataset_multiconfig["two_specie"]["kfold"]["n_structure"],
        ),
        pair_feature_by_seko_method,
        rtol=1e-8,
    )


def test_spin_featurizer(
    model_params_multiconfig,
    pymatgen_structures_multiconfig,
    spin_energy_feature_832,
    spin_force_feature_832,
):
    model_params = model_params_multiconfig["one_specie"]
    pymatgen_structures = pymatgen_structures_multiconfig["one_specie"]

    model_params.use_force = False
    si = SpinFeaturizer(model_params)
    assert round(si(pymatgen_structures[-2:])[-1, 0], 14) == round(
        spin_energy_feature_832, 14
    )

    model_params.use_force = True
    si = SpinFeaturizer(model_params)
    # energy_feature_column_length + force_feature_id
    feature_column_id = 2 + (96 * 1 + 3 * 7 + 2)
    assert round(si(pymatgen_structures[-2:])[feature_column_id, 0], 14) == round(
        spin_force_feature_832, 14
    )
