import numpy as np
import pytest

from para_mlp.data_structure import ModelParams
from para_mlp.featurize import RotationInvariant, SpinFeaturizer
from para_mlp.preprocess import make_force_id


@pytest.mark.parametrize(
    ("sid", "atom_id", "force_comp", "expected"),
    [("00287", 17, 1, 27508), ("00001", 0, 0, 0), ("5000", 31, 2, 479999)],
)
def test_make_force_id(sid, atom_id, force_comp, expected):
    assert make_force_id(sid, atom_id, force_comp) == expected


def test_load_vasp_outputs(dataset, seko_vasprun_outputs, n_atoms_in_structure):
    energy, force, _ = seko_vasprun_outputs
    energy = energy - n_atoms_in_structure * -3.37689

    np.testing.assert_array_equal(
        dataset["target"], np.concatenate((energy, force), axis=0)
    )


def test_split_dataset(divided_dataset, dataset):
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
    model_params, divided_dataset, kfold_feature_by_seko_method
):
    ri = RotationInvariant(model_params)
    np.testing.assert_allclose(
        ri(divided_dataset["kfold"]["structures"]),
        kfold_feature_by_seko_method,
        rtol=1e-8,
    )


def test_spin_featurizer(
    model_params, pymatgen_structures, spin_energy_feature_832, spin_force_feature_832
):
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
