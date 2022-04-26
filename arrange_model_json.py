import json
from pathlib import Path
from typing import Any

import click


@click.command()
@click.option("--gtinv_order", type=int, default=2, show_default=True)
@click.option("--polynomial_model", type=int, default=3, show_default=True)
@click.option("--polynomial_max_order", type=int, default=3, show_default=True)
@click.option("--cutoff_min", type=float, default=6.0, show_default=True)
@click.option("--cutoff_max", type=float, default=10.0, show_default=True)
@click.option("--gaussian_params2_flag", type=int, default=2, show_default=True)
@click.option("--gaussian_params2_num_max", type=int, default=20, show_default=True)
@click.option("--use_force/--no-use_force", default=True, show_default=True)
@click.option("--trial_id_begin", type=int, required=True)
def main(
    gtinv_order,
    polynomial_model,
    polynomial_max_order,
    cutoff_min,
    cutoff_max,
    gaussian_params2_flag,
    gaussian_params2_num_max,
    use_force,
    trial_id_begin,
) -> None:
    """Arrange model.json for machine learning potential generation"""
    defaults_json_path = Path.home() / "para-mlp" / "configs" / "defaults.json"
    with defaults_json_path.open("r") as f:
        defaults_json = json.load(f)
    model_dir = "".join(["model", str(polynomial_model)])

    # Common settings
    defaults_json["polynomial_model"] = polynomial_model
    defaults_json["polynomial_max_order"] = polynomial_max_order
    defaults_json["cutoff_radius_min"] = cutoff_min
    defaults_json["cutoff_radius_max"] = cutoff_max
    defaults_json["gaussian_params2_flag"] = gaussian_params2_flag
    defaults_json["gaussian_params2_num_max"] = gaussian_params2_num_max
    defaults_json["use_force"] = use_force

    gtinv_lmax_list: Any
    if gtinv_order == 2:
        gtinv_lmax_list = [(0,), (4,), (8,)]
    elif gtinv_order == 3:
        gtinv_lmax_list = [
            (0, 0),
            (4, 0),
            (4, 4),
            (8, 0),
            (8, 4),
            (8, 8),
        ]
    elif gtinv_order == 4:
        gtinv_lmax_list = [
            (0, 0, 0),
            (4, 0, 0),
            (4, 4, 0),
            (4, 4, 2),
            (8, 0, 0),
            (8, 4, 0),
            (8, 4, 2),
            (8, 8, 0),
            (8, 8, 2),
        ]

    trial_id = trial_id_begin
    for gtinv_lmax in gtinv_lmax_list:
        defaults_json["gtinv_lmax"] = gtinv_lmax

        trial_dir_path = Path("models") / model_dir / str(trial_id).zfill(3)
        defaults_json["model_dir"] = trial_dir_path.as_posix()

        if not trial_dir_path.exists():
            trial_dir_path.mkdir(parents=True)

        with open("/".join([defaults_json["model_dir"], "model.json"]), "w") as f:
            json.dump(defaults_json, f, indent=4)

        trial_id += 1


if __name__ == "__main__":
    main()
