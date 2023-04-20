import json
import logging
import shutil
from itertools import product
from pathlib import Path
from typing import List, Tuple

import click
from joblib import Parallel, delayed

from para_mlp.preprocess import make_high_energy_struct_dicts

INPUTS_DIR_PATH = Path.home() / "para-mlp" / "data" / "before_augmentation" / "inputs"
PROCESSING_DIR_PATH = (
    Path.home() / "para-mlp" / "data" / "before_augmentation" / "processing"
)


def _edit_model_json(cutoff: float, alpha: float, model_dir_path: Path) -> None:
    """Edit model.json

    Args:
        cutoff (float): The new cutoff radius.
        alpha (float): The new alpha parameter.
        model_dir_path (Path): Path object of model directory.
    """
    model_json_path = model_dir_path / "model.json"
    with model_json_path.open("r") as f:
        model_config_dict = json.load(f)

    model_config_dict["alpha"] = (alpha,)
    model_config_dict["cutoff_radius_max"] = cutoff
    model_config_dict["cutoff_radius_min"] = cutoff

    with model_json_path.open("w") as f:
        json.dump(model_config_dict, f, indent=4)


def dump_modifed_model_json(
    cutoff: float, alpha_exp: int, cnt: int, root_dir_path: Path
) -> None:
    """Modify model.json and dump it

    Args:
        cutoff (float): The new cutoff radius.
        alpha_exp(int): The new alpha exponent.
        cnt (int): The counter of function calls.
        root_dir_path (Path): Path object of root directory.
    """
    alpha = 10 ** (alpha_exp)

    multiple_weight_dir_path = PROCESSING_DIR_PATH / "multiple_weight_source"
    for i in range(24):
        src_dir_id = str(1 + i).zfill(3)
        src_dir_path = multiple_weight_dir_path / "model3" / src_dir_id
        dir_id_begin = 24 * cnt + 1
        dst_dir_id = str(dir_id_begin + i).zfill(3)
        dst_dir_path = root_dir_path / "model3" / dst_dir_id
        shutil.copytree(src_dir_path, dst_dir_path)
        _edit_model_json(cutoff, alpha, model_dir_path=dst_dir_path)

    for i in range(8):
        src_dir_id = str(1 + i).zfill(3)
        src_dir_path = multiple_weight_dir_path / "model4" / src_dir_id
        dir_id_begin = 8 * cnt + 1
        dst_dir_id = str(dir_id_begin + i).zfill(3)
        dst_dir_path = root_dir_path / "model4" / dst_dir_id
        shutil.copytree(src_dir_path, dst_dir_path)
        _edit_model_json(cutoff, alpha, model_dir_path=dst_dir_path)


def _dump_high_energy_struct_dicts(
    model_json_path: Path, high_energy_struct_info: Tuple[str, List[dict]]
) -> None:
    """Dump high_energy_struct_dicts as json

    Args:
        model_json_path (Path): Path of model.json
        high_energy_struct_info (Tuple[str, List[dict]]): The name of sub dataset and
            generated high_energy_struct_dicts about that dataset.
    """
    data_dir_name, high_energy_struct_dict_list = high_energy_struct_info
    data_setting_dir_path = model_json_path.parent / "data_settings" / data_dir_name
    if not data_setting_dir_path.exists():
        data_setting_dir_path.mkdir(parents=True)

    for i, high_energy_struct_dict in enumerate(high_energy_struct_dict_list, 1):
        json_path = data_setting_dir_path / f"high_energy_struct{i}.json"
        with json_path.open("w") as f:
            json.dump(high_energy_struct_dict, f, indent=4)


@click.command()
@click.argument("high_energy_structures_files", nargs=-1)
@click.option(
    "-w",
    "--high_energy_weights",
    default="0.1,0.6",
    show_default=True,
    help="Weights to apply for each high energy structures.",
)
@click.option("--root_dir", required=True, help="Path to root directory.")
@click.option(
    "--data_dir_names",
    default="sqs,fm",
    show_default=True,
    help="Comma separated data dir names.",
)
@click.option(
    "--alpha_exp_min",
    default=-5,
    show_default=True,
    help="The minimum of alpha exponent.",
)
@click.option(
    "--alpha_exp_max",
    default=-2,
    show_default=True,
    help="The maximum of alpha exponent.",
)
@click.option(
    "--one_specie/--no-one_specie",
    default=False,
    show_default=True,
    help="Whether model.json is for one specie potential.",
)
def main(
    high_energy_structures_files,
    high_energy_weights,
    root_dir,
    data_dir_names,
    alpha_exp_min,
    alpha_exp_max,
    one_specie,
) -> None:
    """Arrange model.json for machine learning potential generation (ver. 3)"""
    logging.basicConfig(level=logging.INFO)

    para_mlp_dir_path = Path.home() / "para-mlp"
    model_dir_path = para_mlp_dir_path / "models"
    if one_specie:
        root_dir_path = model_dir_path / "one_specie" / root_dir
    else:
        root_dir_path = model_dir_path / "paramagnetic" / root_dir

    logging.info(" Copying model pool directory")
    logging.info(f"   root_dir:  {root_dir}")
    for cnt, (cutoff, alpha_exp) in enumerate(
        product(range(6, 9), range(alpha_exp_min, alpha_exp_max + 1))
    ):
        dump_modifed_model_json(float(cutoff), alpha_exp, cnt, root_dir_path)

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
    _ = Parallel(n_jobs=-1, verbose=1)(
        delayed(_dump_high_energy_struct_dicts)(
            model_json_path, high_energy_struct_info
        )
        for model_json_path, high_energy_struct_info in model_json_etc
    )

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
