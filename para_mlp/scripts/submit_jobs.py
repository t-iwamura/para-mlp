import os
import subprocess
import time
from pathlib import Path

import click


def make_job_script(name: str) -> str:
    """Make job script from given parameters

    Args:
        name (str): job name

    Returns:
        str: the content of job script file
    """
    job_script_content = (
        "#!/bin/zsh\n"
        f"#SBATCH -J {name}\n"
        "#SBATCH --nodes=1\n"
        "#SBATCH -o std.log\n"
        "#SBATCH -e err.log\n"
        "\n"
        ". ~/.zprofile\n"
        ". ~/.zshrc\n"
        "pyenv activate py39\n"
        "para-mlp train ./model.json"
    )

    return job_script_content


@click.command()
@click.option(
    "--min_id", required=True, type=int, help="minimum id of searching directories"
)
@click.option(
    "--max_id", required=True, type=int, help="maximum id of searching directories"
)
@click.option("--pool_dir_id", required=True, help="model pool directory id")
@click.option(
    "-p", "--partition", default="vega-d", show_default=True, help="partition name"
)
@click.option(
    "--id_digits", type=int, default=3, show_default=True, help="digits filled by zero"
)
def main(min_id, max_id, pool_dir_id, partition, id_digits):
    """Usefull package to submit multiple para-mlp jobs"""
    root_dir_path = Path.cwd()
    inputs_dir_path_list = [
        root_dir_path / str(dir_id).zfill(id_digits)
        for dir_id in range(min_id, max_id + 1)
    ]

    for dir_path in inputs_dir_path_list:
        if not dir_path.exists():
            continue

        job_script_path = dir_path / "job.sh"
        with job_script_path.open("w") as f:
            name = "-".join([pool_dir_id, dir_path.name])
            job_script_content = make_job_script(name)
            f.write(job_script_content)

        os.chdir(dir_path)
        subprocess.call(f"sbatch -p {partition} job.sh", shell=True)
        os.chdir(root_dir_path)

        # wait for safety
        time.sleep(0.1)


if __name__ == "__main__":
    main()
