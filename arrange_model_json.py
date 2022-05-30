import json
from pathlib import Path
from typing import Any, Dict, Tuple

import click


@click.command()
@click.option("--composite_num", type=int, default=2, show_default=True)
@click.option("--gtinv_order", type=int, default=2, show_default=True)
@click.option("--gtinv_lmax_list_type", default="type-a", show_default=True)
@click.option("--polynomial_model", type=int, default=3, show_default=True)
@click.option("--polynomial_max_order", type=int, default=3, show_default=True)
@click.option("--cutoff_min", type=float, default=6.0, show_default=True)
@click.option("--cutoff_max", type=float, default=10.0, show_default=True)
@click.option("--gaussian_params2_flag", type=int, default=2, show_default=True)
@click.option("--gaussian_params2_num_max", type=int, default=20, show_default=True)
@click.option("--use_force/--no-use_force", default=True, show_default=True)
@click.option("--alpha_min_order", type=int, default=3, show_default=True)
@click.option("--alpha_max_order", type=int, default=5, show_default=True)
@click.option("--trial_id_begin", type=int, required=True)
def main(
    composite_num,
    gtinv_order,
    gtinv_lmax_list_type,
    polynomial_model,
    polynomial_max_order,
    cutoff_min,
    cutoff_max,
    gaussian_params2_flag,
    gaussian_params2_num_max,
    use_force,
    alpha_min_order,
    alpha_max_order,
    trial_id_begin,
) -> None:
    """Arrange model.json for machine learning potential generation"""
    if composite_num == 2:
        defaults_json_path = (
            Path.home() / "para-mlp" / "configs" / "two_specie" / "defaults.json"
        )
        model_dir = "/".join(["two_specie", f"model{polynomial_model}"])
    else:
        defaults_json_path = (
            Path.home() / "para-mlp" / "configs" / "one_specie" / "defaults.json"
        )
        model_dir = "/".join(["one_specie", f"model{polynomial_model}"])
    with defaults_json_path.open("r") as f:
        defaults_json = json.load(f)

    # Common settings
    defaults_json["composite_num"] = composite_num
    defaults_json["polynomial_model"] = polynomial_model
    defaults_json["polynomial_max_order"] = polynomial_max_order
    defaults_json["cutoff_radius_min"] = cutoff_min
    defaults_json["cutoff_radius_max"] = cutoff_max
    defaults_json["gaussian_params2_flag"] = gaussian_params2_flag
    defaults_json["gaussian_params2_num_max"] = gaussian_params2_num_max
    defaults_json["use_force"] = use_force

    alpha = tuple(
        10 ** (-order) for order in range(alpha_min_order, alpha_max_order + 1)
    )
    defaults_json["alpha"] = alpha

    gtinv_lmax_list_dict: Dict[Tuple[int, int, str], Any] = {}

    # For composite_num = 1
    gtinv_lmax_list_dict[(1, 2, "type-a")] = [(0,), (4,), (8,)]
    gtinv_lmax_list_dict[(1, 2, "type-b")] = [(2,), (6,)]
    gtinv_lmax_list_dict[(1, 3, "type-a")] = [
        (0, 0),
        (4, 0),
        (4, 4),
        (8, 0),
        (8, 4),
        (8, 8),
    ]
    gtinv_lmax_list_dict[(1, 3, "type-b")] = [
        (2, 0),
        (2, 2),
        (4, 2),
        (6, 0),
        (6, 2),
        (6, 4),
        (6, 6),
        (8, 2),
        (8, 6),
    ]
    gtinv_lmax_list_dict[(1, 4, "type-a")] = [
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
    gtinv_lmax_list_dict[(1, 4, "type-b")] = [
        (2, 0, 0),
        (2, 2, 0),
        (2, 2, 2),
        (4, 2, 0),
        (4, 2, 2),
        (6, 0, 0),
        (6, 2, 0),
        (6, 2, 2),
        (6, 4, 0),
        (6, 4, 2),
        (6, 6, 0),
        (6, 6, 2),
        (8, 2, 0),
        (8, 2, 2),
        (8, 6, 0),
        (8, 6, 2),
    ]

    # For composite_num = 2
    gtinv_lmax_list_dict[(2, 2, "type-a")] = [(0,)]
    gtinv_lmax_list_dict[(2, 2, "type-b")] = [(2,)]
    gtinv_lmax_list_dict[(2, 2, "type-c")] = [(1,)]
    gtinv_lmax_list_dict[(2, 3, "type-a")] = [(0, 0)]
    gtinv_lmax_list_dict[(2, 3, "type-b")] = [
        (4, 0),
        (4, 4),
    ]
    gtinv_lmax_list_dict[(2, 4, "type-a")] = [(0, 0, 0)]
    gtinv_lmax_list_dict[(2, 4, "type-b")] = [
        (4, 0, 0),
        (4, 4, 0),
        (4, 4, 2),
    ]

    trial_id = trial_id_begin
    for gtinv_lmax in gtinv_lmax_list_dict[
        (composite_num, gtinv_order, gtinv_lmax_list_type)
    ]:
        defaults_json["gtinv_lmax"] = gtinv_lmax

        trial_dir_path = Path("models") / model_dir / str(trial_id).zfill(3)
        if not trial_dir_path.exists():
            trial_dir_path.mkdir(parents=True)
        defaults_json["model_dir"] = trial_dir_path.as_posix()

        model_json_path = trial_dir_path / "model.json"
        with model_json_path.open("w") as f:
            json.dump(defaults_json, f, indent=4)

        trial_id += 1


if __name__ == "__main__":
    main()
