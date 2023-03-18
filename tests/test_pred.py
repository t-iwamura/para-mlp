import numpy as np
import pytest

from para_mlp.pred import predict_property


def test_predict_property(inputs_dir_path, outputs_dir_path, structure_ids):
    """Check if predicted properties equal to the outputs of seko's program"""
    structure_file = "/".join([str(inputs_dir_path), "sqs/data/04075/structure.json"])
    model_dir_path = outputs_dir_path / "one_specie"
    predict_dict = predict_property(str(model_dir_path), structure_file)

    force_array = np.array(
        [
            [-2.65683623, -6.36853199, 2.46524136],
            [-0.6447221, -5.485043, 4.35662885],
            [1.6904598, 1.64146033, 1.0282559],
            [0.86055054, -1.51554315, -2.40546428],
            [0.96680229, -2.97876981, -2.61106979],
            [-1.14955235, -3.99704284, 0.1598672],
            [-1.83484515, 3.48488641, 0.97163325],
            [0.97572429, -1.22419541, 3.44292404],
            [-2.3223235, -5.96428915, 3.71713357],
            [-2.70950732, 7.45912326, -4.43808761],
            [-0.92359385, -0.16527205, -1.02756634],
            [3.93914299, 4.1469137, -0.40952368],
            [1.04151777, 0.38736917, -1.75991146],
            [2.40001644, -2.03990858, -0.28208599],
            [4.05100691, 4.18279909, 2.41939916],
            [-3.41473575, 2.85828192, -1.36207206],
            [-2.29939124, 5.31593702, -1.3337171],
            [2.42549057, 2.52876237, 2.29233545],
            [-0.63305082, 6.51478326, 3.02130313],
            [-2.40741072, 1.96824345, -1.3661745],
            [2.26257353, 9.41949059, -5.93447753],
            [3.01365565, -3.35807356, 2.34176547],
            [-4.46415964, 3.69848145, -0.86854098],
            [-4.36926411, -0.32229437, 4.87881784],
            [-2.0427222, -2.9738551, -0.41582752],
            [6.01179922, -1.18146651, -8.25782643],
            [1.53243792, -1.69307137, -3.89263352],
            [-4.62959015, -1.74931209, 2.61651388],
            [1.11690077, -7.26450563, 7.03967283],
            [4.24473858, 3.64730508, -0.57198133],
            [-4.14414987, -3.68977278, -0.76807191],
            [4.11303774, -5.28288971, -3.04645991],
        ]
    )
    force = force_array.flatten()

    assert predict_dict["energy"] == pytest.approx(-127.4489267488, rel=1e-10)
    np.testing.assert_allclose(predict_dict["force"], force, rtol=3.1e-4)

    # For multicomponent potential
    test_structure_files = [
        "/".join([str(inputs_dir_path), "sqs/data", structure_id, "structure.json"])
        for structure_id in structure_ids[:10]
    ]
    model_dir_path = outputs_dir_path / "two_specie"
    test_structure_energies = [
        predict_property(str(model_dir_path), structure_file)["energy"]
        for structure_file in test_structure_files
    ]
    expected_structure_energies = [
        -3.9704145111,
        -4.2398200442,
        -4.6697206919,
        -4.8051147478,
        -4.7611643028,
        -3.6937663095,
        -3.8158463846,
        -4.7933153374,
        -4.3076555293,
        -4.8023456726,
    ]
    expected_structure_energies = [
        energy * 32 for energy in expected_structure_energies
    ]  # '32' is the number of atoms in each structure
    np.testing.assert_allclose(
        test_structure_energies, expected_structure_energies, rtol=1e-9
    )
