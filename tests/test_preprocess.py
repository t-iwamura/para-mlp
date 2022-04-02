import numpy as np
import pytest

from para_mlp.featurize import RotationInvariant, SpinFeaturizer
from para_mlp.preprocess import make_force_id


@pytest.mark.parametrize(
    ("sid", "atom_id", "force_comp", "expected"),
    [("00287", 17, 1, 27508), ("00001", 0, 0, 0), ("5000", 31, 2, 479999)],
)
def test_make_force_id(sid, atom_id, force_comp, expected):
    assert make_force_id(sid, atom_id, force_comp) == expected


def test_load_vasp_outputs(dataset, seko_vasprun_outputs):
    energy, force, _ = seko_vasprun_outputs
    np.testing.assert_array_equal(
        dataset["energy"],
        energy,
    )
    np.testing.assert_array_equal(
        dataset["force"],
        force,
    )


def test_split_dataset(divided_dataset, dataset):
    np.testing.assert_array_equal(
        np.concatenate(
            (divided_dataset["test"]["target"], divided_dataset["kfold"]["target"]),
            axis=0,
        ),
        np.concatenate((dataset["energy"], dataset["force"]), axis=0),
    )


def test_rotation_invariant(
    model_params, divided_dataset, kfold_feature_by_seko_method
):
    ri = RotationInvariant(model_params)
    np.testing.assert_allclose(
        ri(divided_dataset["kfold"]["structures"]),
        kfold_feature_by_seko_method,
        rtol=1e-8,
    )


def test_spin_featurizer(model_params, pymatgen_structures, spin_feature_832):
    si = SpinFeaturizer(model_params)
    assert si(pymatgen_structures[-2:])[-1, 0] == spin_feature_832
