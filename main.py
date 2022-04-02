import logging
from pathlib import Path

import click

from para_mlp.config import load_config
from para_mlp.preprocess import create_dataset, split_dataset
from para_mlp.train import dump_model, train_and_eval


@click.command()
@click.argument("config_path", nargs=1)
def main(config_path):
    """open source package to create paramagnetic machine learning potential"""
    config = load_config(config_path)

    # logger
    if config.save_log:
        log_basename = Path(config_path).stem
        logfile_path = "/".join(["logs", f"{log_basename}.log"])
        logging.basicConfig(filename=logfile_path, level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    logging.info("Preparing dataset")
    dataset = create_dataset(
        config.data_dir, config.targets_json, config.use_force, config.n_jobs
    )
    kfold_dataset, test_dataset = split_dataset(
        dataset, use_force=config.use_force, shuffle=config.shuffle
    )

    logging.info("Training and evaluating")
    best_model, best_model_params = train_and_eval(config, kfold_dataset, test_dataset)

    logging.info("Dumping the best model and its model parameter")
    dump_model(best_model, best_model_params, config.model_dir)
