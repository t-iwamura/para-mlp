import numpy as np
import pytest

from para_mlp.utils import make_high_energy_index, make_yids_for_structure_ids


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


def test_make_high_energy_index(
    high_energy_config,
    yids_for_kfold_high_energy,
    expected_high_energy_index,
):
    high_energy_index = make_high_energy_index(
        high_energy_structure_file_id=1,
        config=high_energy_config,
        n_structure=100,
        force_id_unit=12,
        yids_for_kfold=yids_for_kfold_high_energy,
    )
    np.testing.assert_equal(high_energy_index, expected_high_energy_index)
