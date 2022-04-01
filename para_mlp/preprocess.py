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
    """Dump dataset['energy'] and dataset['force'] as vasp_outputs.json

    Args:
        dataset (Dict[str, Any]): Dataset dict. The keys are 'energy' and 'force'.
        data_dir (str, optional): Path to the data directory where npy files are dumped.
            Defaults to "data/processing".
    """
    energy_npy_path = "/".join([data_dir, "energy"])
    np.save(energy_npy_path, dataset["energy"])

    force_npy_path = "/".join([data_dir, "force"])
    np.save(force_npy_path, dataset["force"])


def make_force_id(sid: str, atom_id: int, force_comp: int) -> int:
    """Make force id from given three ids

    Args:
        sid (str): structure id
        atom_id (int): atom id in structure
        force_comp (int): Force component id of atom.
            The x, y, z component cooresponds to 0, 1, 2.

    Returns:
        int: force id. The id ranges from 0 to [96 * {number of structures used}]
    """
    numerical_sid = int(sid) - 1
    force_id = 96 * numerical_sid + 3 * atom_id + force_comp

    return force_id


def create_dataset(
    data_dir: str, targets_json: str, use_force: bool = False, n_jobs: int = -1
) -> Dict[str, Any]:
    """Create dataset from energy.npy, force.npy and structure.json

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.
        n_jobs (int, optional): Core numbers used. Defaults to -1.

    Raises:
        FileNotFoundError: If the file of targets_json does not exist.

    Returns:
        Dict[str, Any]: Dataset dict. The keys are 'energy', 'force', and 'structures'.
    """
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
        delayed(_load_vasp_jsons)(
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
    """Create dataset by loading vasp_outputs.json and structure.json

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.
        n_jobs (int, optional): Core numbers used. Defaults to 1.

    Raises:
        FileNotFoundError: If the file of targets_json does not exist.

    Returns:
        Dict[str, Any]: Dataset dict
    """
    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    vasprun_pool_path = Path(data_dir) / "inputs" / "data"
    energy, force, structures = zip(
        *Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_load_vasp_jsons)(
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
    """Load vasp outputs, i.e. submatrix of energy.npy and force.npy.

    Args:
        data_dir (str): path to data directory
        structure_ids (Tuple[str]): structure ids used as whole dataset
        use_force (bool, optional): whether to use force of atoms as dataset.
            Defaults to False.

    Raises:
        FileNotFoundError: If energy.npy does not exist
        FileNotFoundError: If force.npy does not exist

    Returns:
        Any: Energy vector of used structures.
            If use_force is True, force vector is also returned.
    """
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


def _load_vasp_jsons(
    vasprun_dir_path: Path, load_vasp_outputs: bool = False, use_force: bool = False
) -> Any:
    """Load vasp_outputs.json and structure.json

    Args:
        vasprun_dir_path (Path): path to directory where jsons of given structure
            is saved
        load_vasp_outputs (bool, optional): Whether to load vasp_outputs.json too,
            not just structure.json. Defaults to False.
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.

    Returns:
        Any: Structure class is returned. If load_vasp_outputs is True,
            energy is also returned. If use_force is True, force is also returned.
    """
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
    """Split given dataset to test dataset and kfold dataset

    Args:
        dataset (Dict[str, Any], optional): Dataset dict. Defaults to None.
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.
        test_size (float, optional): The ratio of test dataset in whole dataset.
            Defaults to 0.1.
        shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.

    Raises:
        KeyError: If 'force' key is not set in dataset dict

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: test dataset and kfold dataset
    """
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
    """Make tempfile which is read by Vasprun class of seko

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json

    Raises:
        FileNotFoundError: If data_dir/inputs/data directory does not exist
        FileNotFoundError: If targets.json does not exist

    Returns:
        str: The filename of tempfile
    """
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
    """Read vasprun tempfile by seko's Vasprun class

    Args:
        vasprun_tempfile (str): vasprun tempfile's name

    Returns:
        Any: energy, force, and list of structures. The structures are
            instances of seko's original class.
    """
    energy, force, _, seko_structures, _ = ReadVaspruns(vasprun_tempfile).get_data()

    return energy, force, seko_structures


def create_dataset_by_seko_method(
    data_dir: str, targets_json: str, use_force: bool = False
) -> Dict[str, Any]:
    """Create dataset by the method which seko implemented in his own program

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.

    Raises:
        FileNotFoundError: If data directory does not exist
        FileNotFoundError: If targets.json does not exist

    Returns:
        Dict[str, Any]: dataset dict
    """
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
