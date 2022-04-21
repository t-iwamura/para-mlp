import json
import os
from itertools import product
from pathlib import Path

import click


@click.command()
@click.option("--lmax_min", type=int, required=True)
@click.option("--lmax_max", type=int, required=True)
@click.option("--trial_id_begin", type=int, required=True)
def main(lmax_min, lmax_max, trial_id_begin) -> None:
    defaults_no_spin_path = (
        Path.home()
        / "para-mlp"
        / "models"
        / "spin_feature_effect"
        / "defaults_no_spin_feature.json"
    )
    with defaults_no_spin_path.open("r") as f:
        defaults_no_spin = json.load(f)

    trial_id = trial_id_begin
    for i, j in product(range(lmax_min, lmax_max + 1), range(lmax_min, lmax_max + 1)):
        if i >= j:
            defaults_no_spin["gtinv_lmax"] = (i, j)
            model_dir = "/".join(
                [
                    "models/spin_feature_effect",
                    str(trial_id).zfill(3),
                    "no_spin_feature",
                ]
            )
            defaults_no_spin["model_dir"] = model_dir

            if not Path(model_dir).exists():
                os.makedirs(model_dir)

            with open("/".join([model_dir, "model.json"]), "w") as f:
                json.dump(defaults_no_spin, f, indent=4)

            trial_id += 1


if __name__ == "__main__":
    main()
