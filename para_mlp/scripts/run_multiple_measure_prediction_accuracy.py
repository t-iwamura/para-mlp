import json
import logging
from pathlib import Path
from typing import Any, Dict

import click
from tqdm import tqdm

from para_mlp.pred import evaluate_prediction_accuracy_for_group
from para_mlp.preprocess import create_dataset


@click.command()
@click.option("--model_set_dir", required=True, help="Path to model set directory.")
@click.option("--structure_ids_file", required=True, help="Path to structure ids file.")
@click.option("--trial_id", type=int, required=True, help="ID of a prediction trial.")
def main(model_set_dir, structure_ids_file, trial_id) -> None:
    """Search model?/{000-999} within model_set_dir"""
    logging.basicConfig(
        level=logging.INFO, format="{asctime} {name}: {message}", style="{"
    )

    with open(structure_ids_file) as f:
        structure_ids = [line.strip() for line in f]

    data_dir_path = Path.home() / "para-mlp" / "data" / "before_augmentation"
    targets_json_path = Path.home() / "para-mlp" / "configs" / "targets.json"
    original_dataset = create_dataset(
        data_dir=str(data_dir_path), targets_json=str(targets_json_path)
    )

    processing_dir_path = data_dir_path / "processing"
    yid_test_json_path = processing_dir_path / "use_force_too" / "yid_test.json"
    with yid_test_json_path.open("r") as f:
        yids_for_test = json.load(f)

    energy_yids = [int(structure_id) - 1 for structure_id in structure_ids]
    energy_yids_for_test = [
        yid for yid in energy_yids if yid in yids_for_test["energy"]
    ]
    dataset = {
        "structures": [
            original_dataset["structures"][sid] for sid in energy_yids_for_test
        ],
        "energy": original_dataset["target"][energy_yids_for_test],
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

        rmse = evaluate_prediction_accuracy_for_group(model_dir, dataset)

        predict_dict: Dict[str, Any] = {}
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
