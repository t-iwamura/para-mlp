import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_std_log(logfile: str) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Parse std.log

    Args:
        logfile (str): The path to std.log

    Returns:
        Tuple[List[Dict[str, Any]], List[float]]: The model parameters of all the test
            models and scores of the models
    """
    logfile_path = Path(logfile)
    with logfile_path.open("r") as f:
        lines = f.readlines()

    # Start parsing
    block_column_ids = [i for i, line in enumerate(lines) if "Test model" in line]

    models, scores = [], []
    for column_id in block_column_ids:
        model_params = {}
        model_params_list = re.findall(r"'[\w]+': [0-9.]+", lines[column_id + 1])
        key, val = model_params_list[0].split(": ")
        model_params[key.replace("'", "")] = float(val)
        key, val = model_params_list[1].split(": ")
        model_params[key.replace("'", "")] = float(val)
        key, val = model_params_list[2].split(": ")
        model_params[key.replace("'", "")] = int(val)
        models.append(model_params)

        score_match = re.search(r"[0-9.]+\n", lines[column_id + 5])
        score = float(score_match.group().strip())
        scores.append(score)

    return models, scores
