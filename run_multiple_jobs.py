import os
import subprocess
import sys
from pathlib import Path
from typing import Generator

import click


def _get_stdout_lines(model_json: str) -> Generator:
    proc = subprocess.Popen(
        ["para-mlp", model_json],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    while True:
        line = proc.stdout.readline()
        if line:
            yield line

        if proc.poll() is not None:
            break


def run_para_mlp(model_dir_name: str) -> None:
    log_dir = Path("logs") / model_dir_name
    if not log_dir.is_dir():
        os.makedirs(log_dir.as_posix())
    std_log_path = log_dir / "std.log"

    model_json_path = Path("models") / model_dir_name / "model.json"

    separator = "DEBUG:para_mlp.train: Test model"
    # Stream stdout(and stderr) to screen and log file
    for line in _get_stdout_lines(model_json_path.as_posix()):
        # Enter new line to make log output more readable
        if separator in line:
            split_line = line.split(separator)[0]
            print(split_line, file=sys.stdout)
            print(separator, file=sys.stdout)
            with std_log_path.open("a") as f:
                print(split_line, file=f)
                print(separator, file=f)
        else:
            sys.stdout.write(line)
            with std_log_path.open("a") as f:
                f.write(line)


@click.command()
@click.option("--id_max", type=int, required=True)
@click.option("--id_min", type=int, required=True)
def main(id_max, id_min) -> None:
    trial_ids = tuple(str(i).zfill(3) for i in range(id_min, id_max + 1))

    for trial_id in trial_ids:
        # Run job by using feature without spin type feature
        model_dir_name = "/".join(["spin_feature_effect", trial_id, "no_spin_feature"])
        run_para_mlp(model_dir_name)

        # Run job by using feature with spin type feature
        model_dir_name = "/".join(["spin_feature_effect", trial_id, "spin_feature"])
        run_para_mlp(model_dir_name)


if __name__ == "__main__":
    main()
