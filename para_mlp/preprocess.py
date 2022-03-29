import json
import sys
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

import numpy as np
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split

mlp_build_tools_path = (
    Path.home() / "mlp-Fe" / "mlptools" / "mlp_build_tools" / "cpp" / "lib"
)
sys.path.append(mlp_build_tools_path.as_posix())


def create_dataset(
    data_dir: str, targets_json: str, use_force: bool = False
) -> Dict[str, Any]:
    data_dir_path = Path(data_dir)
    if data_dir_path.exists():
        inputs_dir_path = data_dir_path / "inputs" / "data"
    else:
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    structures = []
    energies = []
    forces = []
    for sid in structure_ids:
        # Load vasp_outputs.json
        vasp_outputs_json_path = inputs_dir_path / sid / "vasp_outputs.json"
        with vasp_outputs_json_path.open("r") as f:
            vasp_outputs = json.load(f)

        energies.append(vasp_outputs["energy"])

        if use_force:
            for force_component in chain.from_iterable(vasp_outputs["force"]):
                forces.append(force_component)

        # Load structure.json
        structure_json_path = inputs_dir_path / sid / "structure.json"
        with structure_json_path.open("r") as f:
            structure_dict = json.load(f)
        structure = Structure.from_dict(structure_dict)
        structures.append(structure)

    dataset = {"energy": np.array(energies), "structures": structures}

    if use_force:
        dataset["force"] = np.array(forces)

    return dataset


def split_dataset(
    dataset: Dict[str, Any] = None, test_size: float = 0.1, shuffle: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    structure_train, structure_test, y_train, y_test = train_test_split(
        dataset["structures"], dataset["energy"], test_size=test_size, shuffle=shuffle
    )

    kfold_dataset = {"structures": structure_train, "energy": y_train}
    test_dataset = {"structures": structure_test, "energy": y_test}

    return kfold_dataset, test_dataset


def make_vasprun_tempfile(data_dir: str, targets_json: str) -> str:
    data_dir_path = Path(data_dir)
    if data_dir_path.exists():
        inputs_dir_path = data_dir_path / "inputs" / "data"
        inputs_dir = inputs_dir_path.as_posix()
    else:
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    tempfile_lines = [
        "/".join([inputs_dir, sid, "vasprun.xml_1_type"]) for sid in structure_ids
    ]
    tempfile_content = "\n".join(tempfile_lines)

    temp_object = NamedTemporaryFile(mode="w", delete=False)
    temp_object.write(tempfile_content)
    temp_object.close()

    return temp_object.name


def create_dataset_by_seko_method(data_dir: str, targets_json: str) -> Dict[str, Any]:
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    targets_json_path = Path(targets_json)
    if not targets_json_path.exists():
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    vasprun_tempfile = make_vasprun_tempfile(
        data_dir=data_dir, targets_json=targets_json
    )

    energy, force, stress, seko_structures, volume = ReadVaspruns(
        vasprun_tempfile
    ).get_data()

    structures = [
        Structure(
            struct.get_axis().transpose(),
            struct.get_elements(),
            struct.get_positions().transpose(),
        )
        for struct in seko_structures
    ]

    dataset = {"energy": energy, "structures": structures}

    return dataset
