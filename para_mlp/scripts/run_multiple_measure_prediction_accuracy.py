import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import click
from tqdm import tqdm

from para_mlp.pred import evaluate_prediction_accuracy_for_group
from para_mlp.preprocess import create_dataset
from para_mlp.utils import make_yids_for_structure_ids


@click.command()
@click.option("--model_set_dir", required=True, help="Path to model set directory.")
@click.option("--structure_ids_file", required=True, help="Path to structure ids file.")
@click.option("--trial_id", type=int, required=True, help="ID of a prediction trial.")
@click.option(
    "--data_dir_name",
    default="sqs",
    show_default=True,
    help="The name of data directory.",
)
@click.option(
    "--force/--no_force",
    default=False,
    show_default=True,
    help="Measure force RMSE instead of energy RMSE.",
)
def main(model_set_dir, structure_ids_file, trial_id, data_dir_name, force) -> None:
    """Search model?/{000-999} within model_set_dir"""
    logging.basicConfig(
        level=logging.INFO, format="{asctime} {name}: {message}", style="{"
    )

    if data_dir_name not in structure_ids_file:
        print("The given structure ids file doesn't correspond with data_dir_name.")
        sys.exit(1)

    with open(structure_ids_file) as f:
        structure_ids = [line.strip() for line in f]

    data_dir_path = (
        Path.home()
        / "para-mlp"
        / "data"
        / "before_augmentation"
        / "inputs"
        / data_dir_name
    )
    original_dataset = create_dataset(
        data_dir=str(data_dir_path),
        use_force=force,
    )

    processing_dir_path = data_dir_path / "processing"
    yid_test_json_path = processing_dir_path / "use_force_too" / "yid_test.json"
    with yid_test_json_path.open("r") as f:
        yids_for_test = json.load(f)

    energy_yids = [int(structure_id) - 1 for structure_id in structure_ids]
    energy_yids_for_test = [
        yid for yid in energy_yids if yid in yids_for_test["energy"]
    ]
    if force:
        n_structure = len(original_dataset["structures"])
        force_id_unit = (original_dataset["target"].shape[0] // n_structure) - 1
        force_yids_for_test = make_yids_for_structure_ids(
            energy_yids_for_test, n_structure, force_id_unit, use_force=force
        )["force"]
        dataset = {
            "structures": [
                original_dataset["structures"][sid] for sid in energy_yids_for_test
            ],
            "force": original_dataset["target"][force_yids_for_test],
            "types_list": None,
        }
    else:
        dataset = {
            "structures": [
                original_dataset["structures"][sid] for sid in energy_yids_for_test
            ],
            "energy": original_dataset["target"][energy_yids_for_test],
            "types_list": None,
        }

    model_set_dir_path = Path(model_set_dir)
    model_dir_path_list = [
        model_dir_path
        for model_dir_path in model_set_dir_path.glob("model?/[0-9][0-9][0-9]")
    ]
    for model_dir_path in tqdm(model_dir_path_list):
        model_dir = str(model_dir_path)

        logging.info(" Measurement configuration")
        logging.info(f"     model_dir         : {model_dir}")
        logging.info(f"     structure_ids_file: {structure_ids_file}")

        rmse = evaluate_prediction_accuracy_for_group(
            model_dir, dataset, use_force=force
        )

        predict_dict: Dict[str, Any] = {}
        if force:
            predict_dict["rmse(eV/ang)"] = rmse
        else:
            predict_dict["rmse(meV/atom)"] = rmse
        predict_dict["model_dir"] = model_dir
        predict_dict["structure_ids_file"] = structure_ids_file

        log_dir_path = Path(model_dir.replace("models", "logs"))
        prediction_accuracy_dir_path = log_dir_path / "prediction_accuracy"

        if not prediction_accuracy_dir_path.exists():
            prediction_accuracy_dir_path.mkdir(parents=True)

        accuracy_json_path = (
            prediction_accuracy_dir_path / f"{str(trial_id).zfill(3)}.json"
        )
        with accuracy_json_path.open("w") as f:
            json.dump(predict_dict, f, indent=4)


if __name__ == "__main__":
    main()
