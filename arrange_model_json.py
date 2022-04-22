import json
import os
from itertools import combinations_with_replacement
from pathlib import Path

import click


@click.command()
@click.option("--lmax_min", type=int, required=True)
@click.option("--lmax_max", type=int, required=True)
@click.option("--gtinv_order", type=int, default=2)
@click.option("--model_dir", type=str, default="model3")
@click.option("--trial_id_begin", type=int, required=True)
def main(lmax_min, lmax_max, gtinv_order, model_dir, trial_id_begin) -> None:
    defaults_json_path = Path.home() / "para-mlp" / "configs" / "defaults.json"
    with defaults_json_path.open("r") as f:
        defaults_json = json.load(f)

    lmax_tuple_length = gtinv_order - 1
    trial_id = trial_id_begin
    for lmax_tuple in combinations_with_replacement(
        range(lmax_max, lmax_min - 1, -1), lmax_tuple_length
    ):
        defaults_json["gtinv_lmax"] = lmax_tuple
        trial_dir = "/".join(
            [
                "models",
                model_dir,
                str(trial_id).zfill(3),
            ]
        )
        defaults_json["model_dir"] = trial_dir

        if not Path(trial_dir).exists():
            os.makedirs(trial_dir)

        with open("/".join([trial_dir, "model.json"]), "w") as f:
            json.dump(defaults_json, f, indent=4)

        trial_id += 1


if __name__ == "__main__":
    main()
