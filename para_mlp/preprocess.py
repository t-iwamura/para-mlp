import json
from itertools import chain, product
from pathlib import Path
from random import sample
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure

from para_mlp.utils import make_yids_for_structure_ids


def dump_vasp_outputs(
    dataset: Dict[str, Any], data_dir: str = "data/processing"
) -> None:
    """Dump dataset['energy'] and dataset['force'] as npy binary

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
        Dict[str, Any]: Dataset dict. The keys are 'targets' and 'structures'.
    """
    targets_json_path = Path(targets_json)
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(f"targets_json_path does not exist: {targets_json}")

    dataset = {}

    if use_force:
        energy, force = _load_vasp_outputs(data_dir, structure_ids, use_force)
        dataset["target"] = np.concatenate((energy, force), axis=0)
    else:
        dataset["target"] = _load_vasp_outputs(data_dir, structure_ids, use_force)

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
    atomic_energy: float,
    use_force: bool = False,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Create dataset by loading vasp_outputs.json and structure.json

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json
        atomic_energy (float): isolated atom's energy
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
    energies, force, structures = zip(
        *Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(_load_vasp_jsons)(
                vasprun_pool_path / sid, load_vasp_outputs=True, use_force=True
            )
            for sid in structure_ids
        )
    )

    dataset = {"structures": structures}

    n_atom = len(structures[0].sites)
    cohesive_energy = [energy - n_atom * atomic_energy for energy in energies]
    dataset["energy"] = np.array(cohesive_energy)

    if use_force:
        dataset["force"] = np.array(
            [force_comp for force_comp in chain.from_iterable(force)]
        )

    return dataset


def _load_vasp_outputs(
    data_dir: str, structure_ids: List[str], use_force: bool = False
) -> Any:
    """Load vasp outputs, i.e. submatrix of energy.npy and force.npy.

    Args:
        data_dir (str): path to data directory
        structure_ids (List[str]): structure ids used as whole dataset
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
    dataset: Dict[str, Any],
    use_force: bool = False,
    test_size: float = 0.1,
    shuffle: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Split given dataset to test dataset and kfold dataset

    Args:
        dataset (Dict[str, Any]): Dataset dict to be split.
        use_force (bool, optional): Whether to use force at atoms as dataset.
            Defaults to False.
        test_size (float, optional): The ratio of test dataset in whole dataset.
            Defaults to 0.1.
        shuffle (bool, optional): Whether to shuffle dataset. Defaults to True.

    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
            structure_id and yids to generate test dataset and kfold dataset
    """
    n_structure = len(dataset["structures"])
    force_id_unit = (dataset["target"].shape[0] // n_structure) - 1
    old_sids = [i for i in range(n_structure)]
    if shuffle:
        new_sids = sample(old_sids, k=n_structure)
    else:
        new_sids = old_sids

    test_sid_end = int(n_structure * test_size)
    structure_id = {
        "test": new_sids[:test_sid_end],
        "kfold": new_sids[test_sid_end:],
    }

    yids_for_kfold = make_yids_for_structure_ids(
        structure_id["kfold"], n_structure, force_id_unit, use_force=use_force
    )
    yids_for_test = make_yids_for_structure_ids(
        structure_id["test"], n_structure, force_id_unit, use_force=use_force
    )

    return structure_id, yids_for_kfold, yids_for_test


def dump_ids_for_test_and_kfold(
    structure_id: Dict[str, List[int]],
    yids_for_kfold: Dict[str, List[int]],
    yids_for_test: Dict[str, List[int]],
    processing_dir: str = "data/processing",
    use_force: bool = False,
) -> None:
    """Dump ids which are used to generate test dataset and kfold dataset

    Args:
        structure_id (Dict[str, List[int]]): The ids of structures to designate
            test dataset and kfold dataset. The keys are 'test' and 'kfold'.
        yids_for_kfold (Dict[str, List[int]]): The ids of objective variables to select
            energy and force data in kfold dataset. The keys are 'energy', 'force', and
            'target'.
        yids_for_test (Dict[str, List[int]]): The ids of objective variables to select
            energy and force data in test dataset. The keys are 'energy', 'force', and
            'target'.
        processing_dir (str, optional): Path to processing directory where json files
            are dumped. Defaults to "data/processing".
        use_force (bool, optional): Whether to use force at atoms as dataset.
            Defaults to False.
    """
    if use_force:
        data_dir_path = Path(processing_dir) / "use_force_too"
    else:
        data_dir_path = Path(processing_dir) / "use_energy_only"

    structure_id_path = data_dir_path / "structure_id.json"
    with structure_id_path.open("w") as f:
        json.dump(structure_id, f, indent=4)
    yid_kfold_path = data_dir_path / "yid_kfold.json"
    with yid_kfold_path.open("w") as f:
        json.dump(yids_for_kfold, f, indent=4)
    yid_test_path = data_dir_path / "yid_test.json"
    with yid_test_path.open("w") as f:
        json.dump(yids_for_test, f, indent=4)


def load_ids_for_test_and_kfold(
    processing_dir: str = "data/processing", use_force: bool = False
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Load ids which are used to generate test dataset and kfold dataset

    Args:
        processing_dir (str, optional): Path to data directory whiere json files
            are dumped. Defaults to "data/processing".
        use_force (bool, optional): Whether to use force. Defaults to False.

    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]
            (structure_id, yids_for_kfold, yids_for_test).
            The structure_id is the ids of structures to designate test dataset and
                kfold dataset.
            The yids_for_kfold are the ids of objective variables to designate energy
                and force data in kfold dataset.
            The yids_for_test are the ids of objective variables to designate energy
                and force data in test dataset.
    """
    if use_force:
        data_dir_path = Path(processing_dir) / "use_force_too"
    else:
        data_dir_path = Path(processing_dir) / "use_energy_only"

    structure_id_path = data_dir_path / "structure_id.json"
    with structure_id_path.open("r") as f:
        structure_id = json.load(f)
    yid_kfold_path = data_dir_path / "yid_kfold.json"
    with yid_kfold_path.open("r") as f:
        yids_for_kfold = json.load(f)
    yid_test_path = data_dir_path / "yid_test.json"
    with yid_test_path.open("r") as f:
        yids_for_test = json.load(f)

    return structure_id, yids_for_kfold, yids_for_test


def make_vasprun_tempfile(
    data_dir: str, targets_json: str, composite_num: int = 1
) -> str:
    """Make tempfile which is read by Vasprun class of seko

    Args:
        data_dir (str): path to data directory
        targets_json (str): path to targets.json
        composite_num (int): composite number of constructed potential

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

    vasprun_filename = f"vasprun.xml_{composite_num}_type"

    tempfile_lines = [
        "/".join([inputs_dir, sid, vasprun_filename]) for sid in structure_ids
    ]
    tempfile_content = "\n".join(tempfile_lines)

    temp_object = NamedTemporaryFile(mode="w", delete=False)
    temp_object.write(tempfile_content)
    temp_object.close()

    return temp_object.name


def read_vasprun_tempfile(
    vasprun_tempfile: str, composite_num: int = 1, atomic_energy: float = -3.37689
) -> Any:
    """Read vasprun tempfile by seko's Vasprun class

    Args:
        vasprun_tempfile (str): vasprun tempfile's name
        composite_num (int): composite number of constructed potential
        atomic_energy (float): isolated atom's energy

    Returns:
        Any: energy, force, and list of structures. The structures are
            instances of seko's original class.
    """
    atomic_energies = [atomic_energy for _ in range(composite_num)]
    energy, force, _, seko_structures, _ = ReadVaspruns(
        vasprun_tempfile, n_type=composite_num, atom_e=atomic_energies
    ).get_data()

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
