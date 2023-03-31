import json
import logging
import shutil
import typing
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import click
from tqdm import tqdm

from para_mlp.preprocess import load_ids_for_test_and_kfold
from para_mlp.utils import make_high_energy_yids

INPUTS_DIR_PATH = Path.home() / "para-mlp" / "data" / "before_augmentation" / "inputs"


@typing.no_type_check
def make_global_high_energy_struct_dicts(
    high_energy_struct_dicts: Dict[str, List[Dict[str, Any]]],
    n_all_kfold_structure: int,
    eid_length_dict: Dict[str, int],
    fid_length_dict: Dict[str, int],
) -> Dict[str, List[Dict[str, Any]]]:
    eid_begin, fid_begin = 0, n_all_kfold_structure
    for data_dir_name, high_energy_struct_dict_list in high_energy_struct_dicts.items():
        for high_energy_struct_dict in high_energy_struct_dict_list:
            high_energy_struct_dict["yids"]["energy"] += eid_begin
            high_energy_struct_dict["yids"]["force"] += fid_begin
            high_energy_struct_dict["yids"]["energy"] = high_energy_struct_dict["yids"][
                "energy"
            ].tolist()
            high_energy_struct_dict["yids"]["force"] = high_energy_struct_dict["yids"][
                "force"
            ].tolist()
        eid_begin += eid_length_dict[data_dir_name]
        fid_begin += fid_length_dict[data_dir_name]

    return high_energy_struct_dicts


def make_high_energy_struct_dicts(
    high_energy_structures_files: List[str],
    high_energy_weights: str,
    data_dir_names: str,
) -> Dict[str, List[dict]]:
    """Make the dict about high energy struct dict

    Args:
        high_energy_structures_files (List[str]): List of the path
            to high energy structure file
        high_energy_weights (str): The comma separated weights
            for high energy structures
        data_dir_names (str): The comma separated sub dataset names

    Returns:
        Dict[str, List[dict]]: The dict which receives sub dataset name and
            returns high energy struct dict.
    """
    n_all_kfold_structure = 0
    for data_dir_name in data_dir_names.split(","):
        processing_dir_path = INPUTS_DIR_PATH / data_dir_name / "processing"
        structure_id, _, _ = load_ids_for_test_and_kfold(
            processing_dir=str(processing_dir_path),
            use_force=True,
        )
        n_kfold_structure = len(structure_id["kfold"])
        n_all_kfold_structure += n_kfold_structure

    high_energy_structure_file_weight_dict: Dict[str, List[Tuple[str, float]]] = {}
    high_energy_weight_list = [
        float(weight_str) for weight_str in high_energy_weights.split(",")
    ]
    for struct_file, weight in zip(
        high_energy_structures_files, high_energy_weight_list
    ):
        data_dir_name = struct_file.split("/")[-5]
        if data_dir_name not in high_energy_structure_file_weight_dict:
            high_energy_structure_file_weight_dict[data_dir_name] = []
        high_energy_structure_file_weight_dict[data_dir_name].append(
            (struct_file, weight)
        )

    eid_length_dict, fid_length_dict = {}, {}
    high_energy_struct_dicts: Dict[str, List[Dict[str, Any]]] = {}
    for (
        data_dir_name,
        file_and_weights,
    ) in high_energy_structure_file_weight_dict.items():
        processing_dir_path = INPUTS_DIR_PATH / data_dir_name / "processing"
        structure_id, yids_for_kfold, _ = load_ids_for_test_and_kfold(
            processing_dir=str(processing_dir_path),
            use_force=True,
        )
        n_kfold_structure = len(structure_id["kfold"])

        if data_dir_name not in high_energy_struct_dicts:
            high_energy_struct_dicts[data_dir_name] = []
            eid_length_dict[data_dir_name] = n_kfold_structure
            fid_length_dict[data_dir_name] = len(yids_for_kfold["force"])

        force_id_unit = len(yids_for_kfold["force"]) // n_kfold_structure
        n_structure = n_kfold_structure + len(structure_id["test"])
        for struct_file, weight in file_and_weights:
            with open(struct_file) as f:
                high_energy_sids = [int(struct_id) - 1 for struct_id in f]
            high_energy_yids = make_high_energy_yids(
                high_energy_sids,
                n_structure,
                force_id_unit,
                yids_for_kfold,
                use_force=True,
            )
            high_energy_struct_dict = {
                "yids": high_energy_yids,
                "weight": weight,
                "src_file": struct_file,
            }
            high_energy_struct_dicts[data_dir_name].append(high_energy_struct_dict)

    high_energy_struct_dicts = make_global_high_energy_struct_dicts(
        high_energy_struct_dicts=high_energy_struct_dicts,
        n_all_kfold_structure=n_all_kfold_structure,
        eid_length_dict=eid_length_dict,
        fid_length_dict=fid_length_dict,
    )

    return high_energy_struct_dicts


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
