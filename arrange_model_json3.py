import json
import logging
import shutil
from itertools import product
from pathlib import Path

import click
from tqdm import tqdm

from para_mlp.preprocess import make_high_energy_struct_dicts

INPUTS_DIR_PATH = Path.home() / "para-mlp" / "data" / "before_augmentation" / "inputs"


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
    "--data_dir_names",
    default="sqs,fm",
    show_default=True,
    help="Comma separated data dir names.",
)
@click.option(
    "--one_specie/--no-one_specie",
    default=False,
    show_default=True,
    help="Whether model.json is for one specie potential.",
)
def main(
    high_energy_structures_files,
    root_dir,
    data_dir_names,
    high_energy_weights,
    one_specie,
) -> None:
    """Arrange model.json for machine learning potential generation (ver. 3)"""
    logging.basicConfig(level=logging.INFO)

    para_mlp_dir_path = Path.home() / "para-mlp"
    processing_dir_path = (
        para_mlp_dir_path / "data" / "before_augmentation" / "processing"
    )
    multiple_weight_dir_path = processing_dir_path / "multiple_weight_cutoff_alpha"

    model_dir_path = para_mlp_dir_path / "models"
    if one_specie:
        root_dir_path = model_dir_path / "one_specie" / root_dir
    else:
        root_dir_path = model_dir_path / "paramagnetic" / root_dir
    logging.info(" Copying model pool directory")
    logging.info(f"   root_dir:  {root_dir}")
    shutil.copytree(multiple_weight_dir_path, root_dir_path)

    logging.info(" Arranging a dict about high energy structures")
    high_energy_struct_dicts = make_high_energy_struct_dicts(
        high_energy_structures_files, high_energy_weights, data_dir_names
    )

    logging.info(" Dumping high_energy_struct?.json")
    model_json_path_list = [
        file_path for file_path in root_dir_path.glob("**/model.json")
    ]
    model_json_etc = [
        (model_json_path, high_energy_struct_info)
        for model_json_path, high_energy_struct_info in product(
            model_json_path_list, high_energy_struct_dicts.items()
        )
    ]
    for model_json_path, (data_dir_name, high_energy_struct_dict_list) in tqdm(
        model_json_etc
    ):
        data_setting_dir_path = model_json_path.parent / "data_settings" / data_dir_name
        if not data_setting_dir_path.exists():
            data_setting_dir_path.mkdir(parents=True)

        for i, high_energy_struct_dict in enumerate(high_energy_struct_dict_list, 1):
            json_path = data_setting_dir_path / f"high_energy_struct{i}.json"
            with json_path.open("w") as f:
                json.dump(high_energy_struct_dict, f, indent=4)

    logging.info(" Fixing model.jsons")
    data_dir_list = tuple(
        str(INPUTS_DIR_PATH / data_dir_name)
        for data_dir_name in data_dir_names.split(",")
    )
    for model_json_path in model_json_path_list:
        with model_json_path.open("r") as f:
            model_config_dict = json.load(f)

        model_config_dict["data_dir_list"] = data_dir_list
        model_config_dict["model_dir"] = str(model_json_path.parent)

        if one_specie:
            model_config_dict["composite_num"] = 1
            model_config_dict["is_paramagnetic"] = False

        with model_json_path.open("w") as f:
            json.dump(model_config_dict, f, indent=4)


if __name__ == "__main__":
    main()
