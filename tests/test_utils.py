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
