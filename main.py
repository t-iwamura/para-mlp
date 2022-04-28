import json
import logging
from glob import glob
from pathlib import Path

import click

from para_mlp.analyse import search_pareto_optimal
from para_mlp.config import load_config
from para_mlp.model import dump_model_as_lammps
from para_mlp.pred import predict_property
from para_mlp.preprocess import (
    create_dataset,
    load_ids_for_test_and_kfold,
    split_dataset,
)
from para_mlp.train import train_and_eval


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
    dataset = create_dataset(
        config.data_dir, config.targets_json, config.use_force, config.n_jobs
    )
    if config.use_cache_to_split_data:
        structure_id, yids_for_kfold, yids_for_test = load_ids_for_test_and_kfold(
            processing_dir="data/processing", use_force=config.use_force
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

    logger.info(" Training and evaluating")
    best_model = train_and_eval(config, kfold_dataset, test_dataset)

    logger.info(" Dumping best model and parameters")
    best_model.dump_model(config.model_dir)
    dump_model_as_lammps(best_model, config.model_dir)


@main.command()
@click.option("--model_dir", required=True, help="path to model directory.")
@click.option("--structure_file", required=True, help="path to structure.json.")
def predict(model_dir, structure_file):
    """predict energy and force by machine learning potential"""
    logging.basicConfig(level=logging.INFO)

    logging.info(" Start prediction")
    logging.info(f"     model_dir     : {model_dir}")
    logging.info(f"     structure_file: {structure_file}")

    predict_dict = {"model_dir": model_dir, "structure_file": structure_file}
    predict_dict.update(predict_property(model_dir, structure_file))

    logging.info(" Finished prediction")

    log_dir = model_dir.replace("models", "logs")
    log_dir_path = Path(log_dir)
    if not log_dir_path.exists():
        log_dir_path.mkdir(parents=True)

    logging.info(" Dumping predict.json")

    predict_json_path = Path(log_dir) / "predict.json"
    with predict_json_path.open("w") as f:
        json.dump(predict_dict, f, indent=4)


@main.command()
@click.argument("search_dir")
@click.option(
    "--metric", default="energy", help="metric to choose pareto optimal potentials."
)
@click.option(
    "--outputs_dir",
    default="data/outputs/pareto_optimal_search",
    help="path to outputs directory.",
)
def pareto(search_dir, metric, outputs_dir):
    """search pareto optimal potentials"""
    logging.basicConfig(level=logging.INFO)

    pattern = "/".join([outputs_dir, "[0-9][0-9]"])
    trial_dirs = glob(pattern)
    trial_id = len(trial_dirs) + 1

    output_dir_path = Path(outputs_dir) / str(trial_id).zfill(2)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    calc_info_dict = search_pareto_optimal(search_dir, metric)

    logging.info(" Dumping calculation results")

    pareto_search_json_path = output_dir_path / "pareto_search.json"
    with pareto_search_json_path.open("w") as f:
        json.dump(calc_info_dict, f, indent=4)
