import logging
import os
from pathlib import Path

import click

from para_mlp.config import load_config
from para_mlp.model import dump_model, dump_model_as_lammps
from para_mlp.preprocess import (
    create_dataset,
    load_ids_for_test_and_kfold,
    split_dataset,
)
from para_mlp.train import train_and_eval


@click.command()
@click.argument("config_file", nargs=1)
def main(config_file):
    """open source package to create paramagnetic machine learning potential"""
    config = load_config(config_file)

    # logger
    if config.save_log:
        log_basename = Path(config_file).stem
        log_dir = config.model_dir.replace("models", "logs")
        if not Path(log_dir).exists():
            os.makedirs(log_dir)
        logfile_path = "/".join([log_dir, f"{log_basename}.log"])

        logging.basicConfig(
            level=logging.DEBUG,
            handlers=[logging.FileHandler(logfile_path), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=logging.DEBUG)

    logging.info(" Preparing dataset")
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

    logging.info(" Training and evaluating")
    best_model, best_model_params = train_and_eval(config, kfold_dataset, test_dataset)

    logging.info(" Dumping best model and parameters")
    dump_model(best_model, best_model_params, config.model_dir)
    dump_model_as_lammps(best_model, config.model_dir)
