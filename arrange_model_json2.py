import json
import shutil
from itertools import product
from pathlib import Path

import click


@click.command()
@click.argument("high_energy_structures_files", nargs=-1)
@click.option("--root_dir", required=True, help="Path to root directory.")
@click.option(
    "-w",
    "--high_energy_weights",
    default="0.1,0.6",
    show_default=True,
    help="Weights to apply for each high energy structures.",
)
@click.option(
    "--cutoff", default=8.0, show_default=True, help="The cutoff radius for models."
)
@click.option(
    "--alpha/--no-alpha",
    default=False,
    help="Whether models with a different alpha is generated or not.",
)
@click.option(
    "--one_specie/--no-one_specie",
    default=False,
    help="Whether model.json is for one specie potential.",
)
def main(
    high_energy_structures_files,
    root_dir,
    high_energy_weights,
    cutoff,
    alpha,
    one_specie,
) -> None:
    para_mlp_dir_path = Path.home() / "para-mlp"
    processing_dir_path = (
        para_mlp_dir_path / "data" / "before_augmentation" / "processing"
    )
    if alpha:
        multiple_weight_dir_path = processing_dir_path / "multiple_weight_alpha"
    else:
        multiple_weight_dir_path = processing_dir_path / "multiple_weight"

    model_dir_path = para_mlp_dir_path / "models"
    if one_specie:
        root_dir_path = model_dir_path / "one_specie" / root_dir
    else:
        root_dir_path = model_dir_path / "paramagnetic" / root_dir
    shutil.copytree(multiple_weight_dir_path, root_dir_path)

    model_json_path_list = [
        file_path for file_path in root_dir_path.glob("**/model.json")
    ]
    for model_json_path, (i, high_energy_structures) in product(
        model_json_path_list, enumerate(high_energy_structures_files, 1)
    ):
        src_file_path = Path(high_energy_structures)
        dest_file_path = model_json_path.parent / f"high_energy_structures{i}"
        shutil.copy(src_file_path, dest_file_path)

    for model_json_path in model_json_path_list:
        with model_json_path.open("r") as f:
            model_config_dict = json.load(f)

        model_config_dict["high_energy_weights"] = [
            float(high_energy_weight)
            for high_energy_weight in high_energy_weights.split(",")
        ]
        model_config_dict["cutoff_radius_min"] = cutoff
        model_config_dict["cutoff_radius_max"] = cutoff
        model_config_dict["model_dir"] = str(model_json_path.parent)

        if one_specie:
            model_config_dict["composite_num"] = 1
            model_config_dict["is_paramagnetic"] = False

        with model_json_path.open("w") as f:
            json.dump(model_config_dict, f, indent=4)


if __name__ == "__main__":
    main()
