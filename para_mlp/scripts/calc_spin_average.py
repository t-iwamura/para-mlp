import json
import logging
from pathlib import Path

import click

from para_mlp.pred import calc_spin_average


@click.command()
@click.option("--model_dir", required=True, help="path to model directory.")
@click.option("--structure_file", required=True, help="path to structure.json.")
@click.option(
    "--output_dir",
    default="log",
    show_default=True,
    help="path to output directory where predict.json is dumped",
)
def main(model_dir, structure_file, output_dir):
    """
    Calculate total energy average over all the spin configurations for given structure
    """
    logging.basicConfig(level=logging.INFO)

    logging.info(" Start calculation")
    logging.info(f"     model_dir     : {model_dir}")
    logging.info(f"     structure_file: {structure_file}")

    predict_dict = calc_spin_average(model_dir, structure_file)

    logging.info(" Finished calculation")

    if output_dir == "log":
        output_dir = model_dir.replace("models", "logs")
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    logging.info(" Dumping average.json")

    predict_json_path = output_dir_path / "average.json"
    with predict_json_path.open("w") as f:
        json.dump(predict_dict, f, indent=4)
