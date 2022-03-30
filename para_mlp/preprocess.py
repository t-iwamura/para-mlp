import json
import sys
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

import numpy as np
from joblib import Parallel, delayed
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split

mlp_build_tools_path = (
    Path.home() / "mlp-Fe" / "mlptools" / "mlp_build_tools" / "cpp" / "lib"
)
sys.path.append(mlp_build_tools_path.as_posix())


def dump_vasp_outputs(
    dataset: Dict[str, Any], data_dir: str = "data/processing"
) -> None:
    energy_npy_path = "/".join([data_dir, "energy"])
    np.save(energy_npy_path, dataset["energy"])


def create_dataset(
    data_dir: str, targets_json: str, use_force: bool = False, n_jobs: int = -1
) -> Dict[str, Any]:
    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    energy_npy_path = Path(data_dir) / "processing" / "energy.npy"
    if energy_npy_path.exists():
        energy = np.load(energy_npy_path.as_posix())
    else:
        raise FileNotFoundError(f"energy_npy_path does not exist: {energy_npy_path}")

    vasprun_pool_path = Path(data_dir) / "inputs" / "data"
    structures = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_load_vasprun)(
            vasprun_pool_path / sid, load_vasp_outputs=False, use_force=False
        )
        for sid in structure_ids
    )

    structure_ids = [int(sid) - 1 for sid in structure_ids]
    dataset = {"energy": energy[structure_ids], "structures": structures}

    if use_force:
        pass

    return dataset


def create_dataset_from_json(
    data_dir: str,
    targets_json: str,
    use_force: bool = False,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    vasprun_pool_path = Path(data_dir) / "inputs" / "data"
    structures, energy, force = zip(
        *Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_load_vasprun)(
                vasprun_pool_path / sid, load_vasp_outputs=True, use_force=True
            )
            for sid in structure_ids
        )
    )

    dataset = {"energy": np.array(energy), "structure": structures}
    if use_force:
        dataset["force"] = np.array(chain.from_iterable(force))

    return dataset


def _load_vasprun(
    vasprun_dir_path: Path, load_vasp_outputs: bool = False, use_force: bool = False
):
    structure_json_path = vasprun_dir_path / "structure.json"
    with structure_json_path.open("r") as f:
        structure_dict = json.load(f)
    structure = Structure.from_dict(structure_dict)

    if load_vasp_outputs:
        vasp_outputs_json_path = vasprun_dir_path / "vasp_outputs.json"
        with vasp_outputs_json_path.open("r") as f:
            vasp_outputs = json.load(f)

        if use_force:
            force = list(force_component for force_component in vasp_outputs["force"])
            return vasp_outputs["energy"], force, structure
        else:
            return vasp_outputs["energy"], structure

    return structure


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
