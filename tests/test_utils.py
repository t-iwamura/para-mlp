import numpy as np
import pytest

from para_mlp.utils import make_yids_for_structure_ids


@pytest.mark.parametrize(
    ("structure_id", "energy_id_length", "force_id_unit", "expected_yids"),
    [
        (
            [3, 7, 1],
            10,
            6,
            {
                "energy": [3, 7, 1],
                "force": [
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                ],
                "target": [
                    3,
                    7,
                    1,
                    28,
                    29,
                    30,
                    31,
                    32,
                    33,
                    52,
                    53,
                    54,
                    55,
                    56,
                    57,
                    16,
                    17,
                    18,
                    19,
                    20,
                    21,
                ],
            },
        )
    ],
)
def test_make_yids_for_structure_ids(
    structure_id, energy_id_length, force_id_unit, expected_yids
):
    yids = make_yids_for_structure_ids(
        structure_id, energy_id_length, force_id_unit, use_force=True
    )
    assert yids == expected_yids


def test_sample_weight_calculator(
    sample_weight_calculator, yids_for_train_sample_weight, expected_sample_weight
):
    sample_weight = sample_weight_calculator.make_sample_weight(
        yids_for_train_sample_weight, n_energy_data=72
    )
    np.testing.assert_allclose(
        sample_weight,
        expected_sample_weight["high_energy"],
    )

    sample_weight_calculator._energy_weight = 2.0
    sample_weight = sample_weight_calculator.make_sample_weight(
        yids_for_train_sample_weight, n_energy_data=72
    )
    np.testing.assert_allclose(
        sample_weight,
        expected_sample_weight["energy"],
    )

    sample_weight_calculator._force_weight = 5.0
    sample_weight = sample_weight_calculator.make_sample_weight(
        yids_for_train_sample_weight, n_energy_data=72
    )
    np.testing.assert_allclose(
        sample_weight,
        expected_sample_weight["force"],
    )
