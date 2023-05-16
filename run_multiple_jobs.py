import re
import subprocess
import sys
from pathlib import Path
from typing import Generator

import click

EXP_DIR_PATH = Path.home() / "para-mlp" / "exp"


def _get_stdout_lines_from_para_mlp(model_json: str) -> Generator:
    """Get the lines of standard output

    Args:
        model_json (str): The path to model.json

    Yields:
        Generator: The each line of stdout
    """
    proc = subprocess.Popen(
        ["para-mlp", "train", model_json],
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
    """Run the command 'para-mlp train */model.json'

    Args:
        model_dir_name (str): The directory name under 'models'
            where model.json exists.
    """
    log_dir = Path("logs") / model_dir_name
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    std_log_path = log_dir / "std.log"

    model_json_path = Path("models") / model_dir_name / "model.json"

    pattern = re.compile(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}:DEBUG:para_mlp.train: Test model"
    )
    # Stream stdout(and stderr) to screen and log file
    for line in _get_stdout_lines_from_para_mlp(model_json_path.as_posix()):
        # Enter new line to make log output more readable
        match = pattern.search(line)
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
@click.option(
    "--trial_id_file",
    default=str(EXP_DIR_PATH / "trial_id"),
    show_default=True,
    help="The path to trial_id file",
)
def main(model_dir, trial_id_file) -> None:
    """Run multiple jobs which use para-mlp package

    \b
    This function reads the following file
        "models/{model_dir}/{trial_id}/model.json"
    and streams stdout to terminal and "std.log".

    \b
    Args:
        model_dir (str): The path to directory where config of jobs are saved.
            The parent directory of trial directories. The child directory of 'models'.
    """
    with open(trial_id_file) as f:
        trial_ids = [line.strip() for line in f]

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
