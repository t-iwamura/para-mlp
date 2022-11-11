import json
import logging
from itertools import product
from pathlib import Path

import click
from pymatgen.core import Structure

from para_mlp.pred import predict_property


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

    with open(structure_file) as f:
        structure_dict = json.load(f)
    structure = Structure.from_dict(structure_dict)
    n_atom = structure.frac_coords.shape[0]

    energy, calc_time = 0.0, 0.0
    for types in product(range(0, 2), repeat=n_atom):
        types_list = [list(types)]
        predict_dict = predict_property(
            model_dir, structure_file, types_list, use_force=False
        )
        energy += predict_dict["energy"]
        calc_time += predict_dict["calc_time"]

    predict_dict["energy"] = energy / (2**n_atom)
    predict_dict["calc_time"] = calc_time / (2**n_atom)
    predict_dict["model_dir"] = model_dir
    predict_dict["structure_file"] = structure_file

    logging.info(" Finished calculation")

    if output_dir == "log":
        output_dir = model_dir.replace("models", "logs")
    output_dir_path = Path(output_dir)
    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)

    logging.info(" Dumping predict.json")

    predict_json_path = output_dir_path / "predict.json"
    with predict_json_path.open("w") as f:
        json.dump(predict_dict, f, indent=4)
