import json
import sys
from itertools import chain, product
from pathlib import Path
from random import sample
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

import numpy as np
from joblib import Parallel, delayed
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure

mlp_build_tools_path = (
    Path.home() / "mlp-Fe" / "mlptools" / "mlp_build_tools" / "cpp" / "lib"
)
sys.path.append(mlp_build_tools_path.as_posix())


def dump_vasp_outputs(
    dataset: Dict[str, Any], data_dir: str = "data/processing"
) -> None:
    energy_npy_path = "/".join([data_dir, "energy"])
    np.save(energy_npy_path, dataset["energy"])

    force_npy_path = "/".join([data_dir, "force"])
    np.save(force_npy_path, dataset["force"])


def make_force_id(sid: str, atom_id: int, force_comp: int) -> int:
    numerical_sid = int(sid) - 1
    force_id = 96 * numerical_sid + 3 * atom_id + force_comp

    return force_id


def create_dataset(
    data_dir: str, targets_json: str, use_force: bool = False, n_jobs: int = -1
) -> Dict[str, Any]:
    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    dataset = {}

    if use_force:
        dataset["energy"], dataset["force"] = _load_vasp_outputs(
            data_dir, structure_ids, use_force
        )
    else:
        dataset["energy"] = _load_vasp_outputs(data_dir, structure_ids, use_force)

    vasprun_pool_path = Path(data_dir) / "inputs" / "data"
    structures = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_load_vasprun_jsons)(
            vasprun_pool_path / sid, load_vasp_outputs=False, use_force=False
        )
        for sid in structure_ids
    )
    dataset["structures"] = structures

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
    energy, force, structures = zip(
        *Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_load_vasprun_jsons)(
                vasprun_pool_path / sid, load_vasp_outputs=True, use_force=True
            )
            for sid in structure_ids
        )
    )

    dataset = {"energy": np.array(energy), "structures": structures}
    if use_force:
        dataset["force"] = np.array(
            [force_comp for force_comp in chain.from_iterable(force)]
        )

    return dataset


def _load_vasp_outputs(
    data_dir: str, structure_ids: Tuple[str], use_force: bool = False
) -> Any:
    processing_dir_path = Path(data_dir) / "processing"
    energy_npy_path = processing_dir_path / "energy.npy"
    if energy_npy_path.exists():
        energy = np.load(energy_npy_path.as_posix())
        energy_ids = [int(sid) - 1 for sid in structure_ids]
    else:
        raise FileNotFoundError(f"energy_npy_path does not exist: {energy_npy_path}")

    if use_force:
        force_npy_path = Path(processing_dir_path) / "force.npy"
        if force_npy_path.exists():
            force = np.load(force_npy_path.as_posix(), allow_pickle=True)
            force_ids = [
                make_force_id(sid, atom_id, force_comp)
                for sid, atom_id, force_comp in product(
                    structure_ids, range(32), range(3)
                )
            ]
            return energy[energy_ids], force[force_ids]
        else:
            raise FileNotFoundError(f"force_npy_path does not exist: {force_npy_path}")
    else:
        return energy[energy_ids]


def _load_vasprun_jsons(
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
            force = list(
                force_comp for force_comp in chain.from_iterable(vasp_outputs["force"])
            )
            return vasp_outputs["energy"], force, structure
        else:
            return vasp_outputs["energy"], structure

    return structure


def split_dataset(
    dataset: Dict[str, Any] = None,
    use_force: bool = False,
    test_size: float = 0.1,
    shuffle: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if use_force:
        if "force" not in dataset.keys():
            raise KeyError("force key does not exist in dataset.")
        else:
            y = np.concatenate((dataset["energy"], dataset["force"]), axis=0)
    else:
        y = dataset["energy"]

    structures = dataset["structures"]
    n_structure = len(structures)
    old_sids = [i for i in range(n_structure)]
    if shuffle:
        new_sids = sample(old_sids, k=n_structure)
    else:
        new_sids = old_sids
    test_sid_end = int(n_structure * test_size)

    y_id_unit = y.shape[0] // n_structure
    yid_for_test_dataset = [
        y_id_unit * sid + yid_per_sid
        for sid, yid_per_sid in product(new_sids[:test_sid_end], range(y_id_unit))
    ]
    test_dataset = {
        "structures": [structures[sid] for sid in new_sids[:test_sid_end]],
        "target": y[yid_for_test_dataset],
    }
    yid_for_kfold_dataset = [
        y_id_unit * sid + yid_per_sid
        for sid, yid_per_sid in product(new_sids[test_sid_end:], range(y_id_unit))
    ]
    kfold_dataset = {
        "structures": [structures[sid] for sid in new_sids[test_sid_end:]],
        "target": y[yid_for_kfold_dataset],
    }

    return test_dataset, kfold_dataset


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


def read_vasprun_tempfile(vasprun_tempfile: str) -> Any:
    energy, force, _, seko_structures, _ = ReadVaspruns(vasprun_tempfile).get_data()

    return energy, force, seko_structures


def create_dataset_by_seko_method(
    data_dir: str, targets_json: str, use_force: bool = False
) -> Dict[str, Any]:
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    targets_json_path = Path(targets_json)
    if not targets_json_path.exists():
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    vasprun_tempfile = make_vasprun_tempfile(
        data_dir=data_dir, targets_json=targets_json
    )

    energy, force, seko_structures = read_vasprun_tempfile(vasprun_tempfile)

    dataset = {"energy": energy}

    if use_force:
        dataset["force"] = force

    structures = [
        Structure(
            struct.get_axis().transpose(),
            struct.get_elements(),
            struct.get_positions().transpose(),
        )
        for struct in seko_structures
    ]

    dataset["structures"] = structures

    return dataset
