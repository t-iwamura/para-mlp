import numpy as np
import pytest

from para_mlp.preprocess import _load_vasp_outputs, make_force_id


@pytest.mark.parametrize(
    ("sid", "atom_id", "force_comp", "expected"),
    [("00287", 17, 1, 27508), ("00001", 0, 0, 0), ("5000", 31, 2, 479999)],
)
def test_make_force_id(sid, atom_id, force_comp, expected):
    assert make_force_id(sid, atom_id, force_comp) == expected


def test_load_vasp_outputs(data_dir, structure_ids, seko_vasprun_outputs):
    energy, force, _ = seko_vasprun_outputs
    np.testing.assert_array_equal(
        _load_vasp_outputs(data_dir, structure_ids, use_force=False),
        energy,
    )
    np.testing.assert_array_equal(
        _load_vasp_outputs(data_dir, structure_ids, use_force=True)[1],
        force,
    )
