from para_mlp.file_IO import parse_std_log


def test_parse_std_log(outputs_dir_path):
    logfile = "/".join([outputs_dir_path.as_posix(), "std.log"])
    models, scores = parse_std_log(logfile)

    expected_models = [
        {"alpha": 0.01, "cutoff_radius": 6.0, "gaussian_params2_num": 10},
        {"alpha": 0.01, "cutoff_radius": 8.0, "gaussian_params2_num": 10},
    ]
    expected_scores = [0.23908108966976513, 0.24386736722057614]

    assert models == expected_models
    assert scores == expected_scores