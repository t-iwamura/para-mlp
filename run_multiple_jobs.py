import re
import subprocess
import sys
from pathlib import Path
from typing import Generator

import click


def _get_stdout_lines_from_para_mlp(model_json: str) -> Generator:
    """Get the lines of standard output

    Args:
        model_json (str): The path to model.json

    Yields:
        Generator: The each line of stdout
    """
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
        elif proc.poll() is not None:
            break


def run_para_mlp(model_dir_name: str) -> None:
    """Run the command 'para-mlp */model.json'

    Args:
        model_dir_name (str): The directory name under 'models'
            where model.json exists.
    """
    log_dir = Path("logs") / model_dir_name
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    std_log_path = log_dir / "std.log"

    model_json_path = Path("models") / model_dir_name / "model.json"

    pattern = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:DEBUG:para_mlp.train: Test model"
    # Stream stdout(and stderr) to screen and log file
    for line in _get_stdout_lines_from_para_mlp(model_json_path.as_posix()):
        # Enter new line to make log output more readable
        match = re.search(pattern, line)
        if match is not None:
            tqdm_loop_string = line.replace(match.group(), "")
            sys.stdout.write(tqdm_loop_string)
            sys.stdout.write(f"{match.group()}\n")
            with std_log_path.open("a") as f:
                f.write(tqdm_loop_string)
                f.write(f"{match.group()}\n")
        else:
            sys.stdout.write(line)
            with std_log_path.open("a") as f:
                f.write(line)


@click.command()
@click.argument("model_dir")
@click.option("--id_max", type=int, required=True)
@click.option("--id_min", type=int, required=True)
def main(model_dir, id_max, id_min) -> None:
    """Run multiple jobs which use para-mlp package

    This function reads the following file
        "models/{model_dir}/{trial_id}/model.json"
    and streams stdout to terminal and "std.log".

    Args:
        model_dir (str): The path to directory where config of jobs are saved.
            The parent directory of trial directories. The child directory of 'models'.
        id_max (int): The maximum of trial id
        id_min (int): The minimum of trial id
    """
    trial_ids = tuple(str(i).zfill(3) for i in range(id_min, id_max + 1))

    for trial_id in trial_ids:
        # Run job by using feature without spin type feature
        model_dir_name = "/".join(
            [
                model_dir,
                trial_id,
            ]
        )
        run_para_mlp(model_dir_name)

        # Run job by using feature with spin type feature
        # model_dir_name = "/".join(["spin_feature_effect", trial_id, "spin_feature"])
        # run_para_mlp(model_dir_name)


if __name__ == "__main__":
    main()
