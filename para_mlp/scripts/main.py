import json
import logging
from glob import glob
from pathlib import Path

import click
import numpy as np

from para_mlp.analyse import search_pareto_optimal
from para_mlp.config import load_config
from para_mlp.model import dump_model_as_lammps
from para_mlp.pred import predict_property
from para_mlp.preprocess import (
    create_dataset,
    load_ids_for_test_and_kfold,
    merge_sub_dataset,
    split_dataset,
)
from para_mlp.train import train_and_eval
from para_mlp.utils import dump_version_info


@click.group()
def main():
    """open source package to create paramagnetic machine learning potential"""
    pass


@main.command()
@click.argument("config_file", nargs=1)
def train(config_file):
    """train machine learning potential"""
    config = load_config(config_file)

    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(name)s:%(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    sh.setFormatter(formatter)

    logger.addHandler(sh)

    if config.save_log:
        log_basename = Path(config_file).stem
        log_dir = config.model_dir.replace("models", "logs")
        log_dir_path = Path(log_dir)
        if not log_dir_path.exists():
            log_dir_path.mkdir(parents=True)
        logfile = "/".join([log_dir, f"{log_basename}.log"])

        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.info(" Preparing dataset")
    all_dataset = {}
    for data_dir in config.data_dir_list:
        dataset = create_dataset(
            data_dir,
            config.use_force,
            config.n_jobs,
        )

        if config.use_cache_to_split_data:
            structure_id, yids_for_kfold, yids_for_test = load_ids_for_test_and_kfold(
                processing_dir="/".join([data_dir, "processing"]),
                use_force=config.use_force,
            )
        else:
            structure_id, yids_for_kfold, yids_for_test = split_dataset(
                dataset, use_force=config.use_force, shuffle=config.shuffle
            )
        kfold_dataset = {
            "structures": [dataset["structures"][sid] for sid in structure_id["kfold"]],
            "target": dataset["target"][yids_for_kfold["target"]],
        }
        test_dataset = {
            "structures": [dataset["structures"][sid] for sid in structure_id["test"]],
            "target": dataset["target"][yids_for_test["target"]],
        }
        if config.composite_num == 2:
            kfold_dataset["types_list"] = [
                dataset["types_list"][sid] for sid in structure_id["kfold"]
            ]
            test_dataset["types_list"] = [
                dataset["types_list"][sid] for sid in structure_id["test"]
            ]
        else:
            kfold_dataset["types_list"] = []
            test_dataset["types_list"] = []

        data_dir_name = data_dir.split("/")[-1]
        all_dataset[data_dir_name] = {
            "kfold": kfold_dataset,
            "test": test_dataset,
        }
    kfold_dataset, test_dataset = merge_sub_dataset(all_dataset, config.data_dir_list)

    logger.info(" Training and evaluating")
    best_model = train_and_eval(config, kfold_dataset, test_dataset)

    logger.info(" Dumping best model and parameters")
    model_dir_path = Path(config.model_dir)
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    best_model.dump_model(config.model_dir)
    dump_model_as_lammps(best_model, config.model_dir)

    dump_version_info(config.model_dir)


@main.command()
@click.option("--model_dir", required=True, help="path to model directory.")
@click.option("--structure_file", required=True, help="path to structure.json.")
@click.option(
    "--repetition",
    type=int,
    default=1,
    show_default=True,
    help="how many times the prediction is repeated.",
)
@click.option(
    "--output_dir",
    default="log",
    show_default=True,
    help="path to output directory where predict.json is dumped",
)
def predict(model_dir, structure_file, repetition, output_dir):
    """predict energy and force by machine learning potential"""
    logging.basicConfig(level=logging.INFO)

    logging.info(" Start prediction")
    logging.info(f"     model_dir     : {model_dir}")
    logging.info(f"     structure_file: {structure_file}")

    energy, force, calc_time = 0.0, np.zeros((96, 1)), 0.0
    for _ in range(repetition):
        predict_dict = predict_property(model_dir, structure_file)
        energy += predict_dict["energy"]
        force += np.reshape(predict_dict["force"], (-1, 1))
        calc_time += predict_dict["calc_time"]

    predict_dict["energy"] = energy / repetition
    force_array = force / repetition
    predict_dict["force"] = force_array.flatten().tolist()
    predict_dict["calc_time"] = calc_time / repetition
    predict_dict["model_dir"] = model_dir
    predict_dict["structure_file"] = structure_file
    predict_dict["repetition"] = repetition

    logging.info(" Finished prediction")

    if output_dir == "log":
        output_dir = model_dir.replace("models", "logs")
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    logging.info(" Dumping predict.json")

    predict_json_path = output_dir_path / "predict.json"
    with predict_json_path.open("w") as f:
        json.dump(predict_dict, f, indent=4)


@main.command()
@click.argument("search_dir")
@click.option(
    "--metric",
    default="energy",
    show_default=True,
    help="metric to choose pareto optimal potentials.",
)
@click.option("--accuracy_file_id", type=int, help="id of accuracy file.")
@click.option(
    "--outputs_dir",
    default="data/outputs/pareto_optimal_search",
    show_default=True,
    help="path to outputs directory.",
)
def pareto(search_dir, metric, accuracy_file_id, outputs_dir):
    """search pareto optimal potentials

    search [001-999]/predict.json within search_dir
    """
    logging.basicConfig(level=logging.INFO)

    pattern = "/".join([outputs_dir, "[0-9][0-9][0-9]"])
    trial_dirs = glob(pattern)
    trial_id = len(trial_dirs) + 1

    output_dir_path = Path(outputs_dir) / str(trial_id).zfill(3)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    accuracy_file_id = str(accuracy_file_id).zfill(3)

    calc_info_dict = search_pareto_optimal(search_dir, metric, accuracy_file_id)

    logging.info(" Dumping calculation results")

    pareto_search_json_path = output_dir_path / "pareto_search.json"
    with pareto_search_json_path.open("w") as f:
        json.dump(calc_info_dict, f, indent=4)
