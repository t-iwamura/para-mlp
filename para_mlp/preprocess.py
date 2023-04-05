import json
import re
from copy import deepcopy
from itertools import chain, product
from pathlib import Path
from random import sample
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Incar, Poscar, Vasprun
from tqdm import tqdm

from para_mlp.utils import make_yids_for_structure_ids


def arrange_structure_jsons(data_dir: str) -> None:
    """Arrange structure.jsons from POSCAR in data_dir

    Args:
        data_dir (str): Path to data directory.
    """
    data_dir_path = Path(data_dir)
    poscar_path_list = [
        p for p in data_dir_path.glob("*/POSCAR") if re.search(r"\d{5}/POSCAR", str(p))
    ]
    for poscar_path in tqdm(poscar_path_list):
        struct_json_path = poscar_path.parent / "structure.json"
        if struct_json_path.exists():
            continue

        poscar = Poscar.from_file(str(poscar_path))

        with struct_json_path.open("w") as f:
            json.dump(poscar.structure.as_dict(), f, indent=4)


def arrange_types_list_jsons(data_root_dir: str) -> None:
    """Arrange types_list.jsons from INCAR and targets.json

    Args:
        data_root_dir (str): Path to data root directory.
    """
    data_root_dir_path = Path(data_root_dir)
    incar_path = data_root_dir_path / "input" / "INCAR"
    incar = Incar.from_file(str(incar_path))

    processing_dir_path = data_root_dir_path / "processing"
    targets_json_path = processing_dir_path / "targets.json"
    with targets_json_path.open("r") as f:
        structure_ids = json.load(f)
    n_structure = len(structure_ids)

    types = [0 if m > 0 else 1 for m in incar["MAGMOM"]]
    types_list = [types for _ in range(n_structure)]

    types_list_json_path = processing_dir_path / "types_list.json"
    with types_list_json_path.open("w") as f:
        json.dump(types_list, f, indent=4)


def arrange_vasp_outputs_jsons(data_dir: str) -> None:
    """Arrange vasp_outputs.jsons from vasprun.xml in data_dir

    Args:
        data_dir (str): Path to data directory.
    """
    data_dir_path = Path(data_dir)
    vasprun_xml_path_list = [p for p in data_dir_path.glob("*/vasprun.xml")]
    _ = Parallel(n_jobs=-1, verbose=1)(
        delayed(_dump_vasp_outputs_json)(vasprun_xml_path)
        for vasprun_xml_path in vasprun_xml_path_list
    )


def _dump_vasp_outputs_json(vasprun_xml_path: Path) -> None:
    """Dump vasp_outputs.json

    Args:
        vasprun_xml_path (Path): Path object about a vasprun.xml.
    """
    vasp_outputs_json_path = vasprun_xml_path.parent / "vasp_outputs.json"
    if vasp_outputs_json_path.exists():
        return None

    vasprun = Vasprun(str(vasprun_xml_path), parse_potcar_file=False)
    vasp_outputs_dict = {
        "energy": vasprun.final_energy,
        "force": vasprun.ionic_steps[-1]["forces"],
    }

    with vasp_outputs_json_path.open("w") as f:
        json.dump(vasp_outputs_dict, f, indent=4)


def dump_vasp_outputs(
    dataset: Dict[str, Any], data_dir: str = "data/processing"
) -> None:
    """Dump dataset['energy'] and dataset['force'] as npy binary

    Args:
        dataset (Dict[str, Any]): Dataset dict. The keys are 'energy' and 'force'.
        data_dir (str, optional): Path to the data directory where npy files are dumped.
            Defaults to "data/processing".
    """
    data_dir_path = Path(data_dir)
    if not data_dir_path.exists():
        data_dir_path.mkdir(parents=True)

    energy_npy_path = data_dir_path / "energy"
    np.save(energy_npy_path, dataset["energy"])

    force_npy_path = data_dir_path / "force"
    np.save(force_npy_path, dataset["force"])


def make_force_id(sid: str, atom_id: int, force_comp: int, n_atom: int) -> int:
    """Make force id from given three ids

    Args:
        sid (str): structure id
        atom_id (int): atom id in structure
        force_comp (int): Force component id of atom.
            The x, y, z component cooresponds to 0, 1, 2.
        n_atom (int): The number of atoms in the structures of given dataset.

    Returns:
        int: force id. The id ranges from 0 to [96 * {number of structures used}]
    """
    numerical_sid = int(sid) - 1
    force_id = (n_atom * 3) * numerical_sid + 3 * atom_id + force_comp

    return force_id


def create_dataset(
    data_dir: str, use_force: bool = False, n_jobs: int = -1
) -> Dict[str, Any]:
    """Create dataset from energy.npy, force.npy and structure.json

    Args:
        data_dir (str): path to data directory
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.
        n_jobs (int, optional): Core numbers used. Defaults to -1.

    Raises:
        FileNotFoundError: If the file of targets_json does not exist.

    Returns:
        Dict[str, Any]: Dataset dict. The keys are 'targets', 'structures'
            and 'types_list'.
    """
    targets_json_path = Path(data_dir) / "processing" / "targets.json"
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(
            f"targets_json_path does not exist: {targets_json_path}"
        )

    dataset = {}

    vasprun_pool_path = Path(data_dir) / "data"
    structures = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_load_vasp_jsons)(
            vasprun_pool_path / sid, load_vasp_outputs=False, use_force=False
        )
        for sid in structure_ids
    )
    dataset["structures"] = structures

    if use_force:
        n_atom = len(dataset["structures"][0].sites)
        types_list, energy, force = _load_vasp_outputs(
            data_dir, structure_ids, use_force, n_atom
        )
        dataset["types_list"] = types_list
        dataset["target"] = np.concatenate((energy, force), axis=0)
    else:
        types_list, energy = _load_vasp_outputs(data_dir, structure_ids, use_force)
        dataset["types_list"] = types_list
        dataset["target"] = energy

    return dataset


def create_dataset_from_json(
    data_dir: str,
    atomic_energy: float,
    use_force: bool = False,
    n_jobs: int = 1,
) -> Dict[str, Any]:
    """Create dataset by loading vasp_outputs.json and structure.json

    Args:
        data_dir (str): path to data directory
        atomic_energy (float): isolated atom's energy
        use_force (bool, optional): Whether to use force of atoms as dataset.
            Defaults to False.
        n_jobs (int, optional): Core numbers used. Defaults to 1.

    Raises:
        FileNotFoundError: If the file of targets_json does not exist.

    Returns:
        Dict[str, Any]: Dataset dict
    """
    targets_json_path = Path(data_dir) / "processing" / "targets.json"
    if targets_json_path.exists():
        with targets_json_path.open("r") as f:
            structure_ids = json.load(f)
    else:
        raise FileNotFoundError(
            f"targets_json_path does not exist: {targets_json_path}"
        )

    vasprun_pool_path = Path(data_dir) / "data"
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
        dataset["target"] = np.concatenate((dataset["energy"], dataset["force"]))
    else:
        dataset["target"] = dataset["energy"].copy()

    return dataset


def _load_vasp_outputs(
    data_dir: str,
    structure_ids: List[str],
    use_force: bool = False,
    n_atom: int = None,
) -> Any:
    """Load vasp outputs, i.e. submatrix of energy.npy and force.npy.

    Args:
        data_dir (str): path to data directory
        structure_ids (List[str]): structure ids used as whole dataset
        use_force (bool, optional): whether to use force of atoms as dataset.
            Defaults to False.
        n_atom (int, optional): the number of atoms in the structures of given dataset.
            Defaults to None.

    Raises:
        FileNotFoundError: If energy.npy does not exist
        FileNotFoundError: If force.npy does not exist

    Returns:
        Any: The list about species and energy vector of used structures.
            If use_force is True, force vector is also returned.
    """
    processing_dir_path = Path(data_dir) / "processing"
    types_list_json_path = processing_dir_path / "types_list.json"
    if types_list_json_path.exists():
        with types_list_json_path.open("r") as f:
            types_list = json.load(f)
        sids = [int(sid) - 1 for sid in structure_ids]
        chosen_types_list = [types_list[sid] for sid in sids]
    else:
        raise FileNotFoundError(
            f"types_list_json_path does not exist: {types_list_json_path}"
        )

    energy_npy_path = processing_dir_path / "energy.npy"
    if energy_npy_path.exists():
        energy = np.load(energy_npy_path)
    else:
        raise FileNotFoundError(f"energy_npy_path does not exist: {energy_npy_path}")

    if use_force:
        force_npy_path = Path(processing_dir_path) / "force.npy"
        if force_npy_path.exists():
            force = np.load(force_npy_path, allow_pickle=True)
            force_ids = [
                make_force_id(sid, atom_id, force_comp, n_atom)
                for sid, atom_id, force_comp in product(
                    structure_ids, range(n_atom), range(3)
                )
            ]
            return chosen_types_list, energy[sids], force[force_ids]
        else:
            raise FileNotFoundError(f"force_npy_path does not exist: {force_npy_path}")
    else:
        return chosen_types_list, energy[sids]


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
    test_ratio: float = 0.1,
    shuffle: bool = True,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Split given dataset to test dataset and kfold dataset

    Args:
        dataset (Dict[str, Any]): Dataset dict to be split.
        use_force (bool, optional): Whether to use force at atoms as dataset.
            Defaults to False.
        test_ratio (float, optional): The ratio of test dataset in whole dataset.
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

    test_sid_end = int(n_structure * test_ratio)
    structure_id = {
        "test": new_sids[:test_sid_end],
        "kfold": new_sids[test_sid_end:],
    }
    structure_id["kfold"].sort()
    structure_id["test"].sort()

    yids_for_kfold = make_yids_for_structure_ids(
        structure_id["kfold"], n_structure, force_id_unit, use_force=use_force
    )
    yids_for_test = make_yids_for_structure_ids(
        structure_id["test"], n_structure, force_id_unit, use_force=use_force
    )

    return structure_id, yids_for_kfold, yids_for_test


def split_dataset_with_addition(
    dataset: Dict[str, Any],
    old_data_dir_name: str,
    use_force: bool = False,
    test_ratio: float = 0.1,
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Split given additioned dataset to test dataset and kfold dataset

    Args:
        dataset (Dict[str, Any]): Dataset dict to be split.
        old_data_dir_name (str): The name of old data directory.
        use_force (bool, optional): Whether or not to use force. Defaults to False.
        test_ratio (float, optional): The ratio of test dataset. Defaults to 0.1.

    Returns:
        Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
            structure_id and yids to generate test dataset and kfold dataset
    """
    para_mlp_dir_path = Path.home() / "para-mlp"
    inputs_dir_path = para_mlp_dir_path / "data" / "before_augmentation" / "inputs"
    processing_dir_path = inputs_dir_path / old_data_dir_name / "processing"
    struct_id_json_path = processing_dir_path / "use_force_too" / "structure_id.json"
    with struct_id_json_path.open("r") as f:
        structure_id = json.load(f)

    n_all_structure = len(dataset["structures"])
    n_old_structure = len(structure_id["test"]) + len(structure_id["kfold"])
    n_new_structure = n_all_structure - n_old_structure

    old_additional_sids = [i for i in range(n_old_structure, n_all_structure)]
    new_additional_sids = sample(old_additional_sids, k=n_new_structure)
    test_sid_end = int(n_new_structure * test_ratio)

    structure_id["kfold"].extend(new_additional_sids[test_sid_end:])
    structure_id["test"].extend(new_additional_sids[:test_sid_end])
    structure_id["kfold"].sort()
    structure_id["test"].sort()

    force_id_unit = (dataset["target"].shape[0] // n_all_structure) - 1
    yids_for_kfold = make_yids_for_structure_ids(
        structure_id["kfold"], n_all_structure, force_id_unit, use_force=use_force
    )
    yids_for_test = make_yids_for_structure_ids(
        structure_id["test"], n_all_structure, force_id_unit, use_force=use_force
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

    if not data_dir_path.exists():
        data_dir_path.mkdir(parents=True)

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
        inputs_dir_path = data_dir_path / "data"
        inputs_dir = str(inputs_dir_path)
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


def merge_sub_dataset(
    all_dataset: Dict[str, dict], data_dir_list: List[str]
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Merge sub datasets into one dataset

    Args:
        all_dataset (Dict[str, dict]): Dict which receives sub dataset name and
            returns a dict about sub dataset.
        data_dir_list (List[str]): List of the path to sub dataset directory.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: The kfold dataset dict and
            test dataset dict.
    """
    kfold_dataset, test_dataset = {}, {}
    for i, data_dir in enumerate(data_dir_list):
        data_dir_name = data_dir.split("/")[-1]
        dataset = all_dataset[data_dir_name]

        n_kfold_structure = len(dataset["kfold"]["structures"])
        n_test_structure = len(dataset["test"]["structures"])
        if i == 0:
            kfold_dataset["structures"] = deepcopy(dataset["kfold"]["structures"])
            kfold_dataset["types_list"] = deepcopy(dataset["kfold"]["types_list"])
            kfold_dataset["energy"] = dataset["kfold"]["target"][
                :n_kfold_structure
            ].copy()
            kfold_dataset["force"] = dataset["kfold"]["target"][
                n_kfold_structure:
            ].copy()
            kfold_dataset["n_structure"] = [len(dataset["kfold"]["structures"])]
            kfold_dataset["n_atom_in_structures"] = [
                len(dataset["kfold"]["structures"][0].sites)
            ]

            test_dataset["structures"] = deepcopy(dataset["test"]["structures"])
            test_dataset["types_list"] = deepcopy(dataset["test"]["types_list"])
            test_dataset["energy"] = dataset["test"]["target"][:n_test_structure].copy()
            test_dataset["force"] = dataset["test"]["target"][n_test_structure:].copy()
            test_dataset["n_structure"] = [len(dataset["test"]["structures"])]
            test_dataset["n_atom_in_structures"] = [
                len(dataset["test"]["structures"][0].sites)
            ]
        else:
            kfold_dataset["structures"].extend(dataset["kfold"]["structures"])
            kfold_dataset["types_list"].extend(dataset["kfold"]["types_list"])
            kfold_dataset["energy"] = np.concatenate(
                (
                    kfold_dataset["energy"],
                    dataset["kfold"]["target"][:n_kfold_structure],
                )
            )
            kfold_dataset["force"] = np.concatenate(
                (kfold_dataset["force"], dataset["kfold"]["target"][n_kfold_structure:])
            )
            kfold_dataset["n_structure"].append(len(dataset["kfold"]["structures"]))
            kfold_dataset["n_atom_in_structures"].append(
                len(dataset["kfold"]["structures"][0].sites)
            )

            test_dataset["structures"].extend(dataset["test"]["structures"])
            test_dataset["types_list"].extend(dataset["test"]["types_list"])
            test_dataset["energy"] = np.concatenate(
                (test_dataset["energy"], dataset["test"]["target"][:n_test_structure])
            )
            test_dataset["force"] = np.concatenate(
                (test_dataset["force"], dataset["test"]["target"][n_test_structure:])
            )
            test_dataset["n_structure"].append(len(dataset["test"]["structures"]))
            test_dataset["n_atom_in_structures"].append(
                len(dataset["test"]["structures"][0].sites)
            )
    kfold_dataset["target"] = np.concatenate(
        (kfold_dataset["energy"], kfold_dataset["force"])
    )
    test_dataset["target"] = np.concatenate(
        (test_dataset["energy"], test_dataset["force"])
    )

    if len(kfold_dataset["types_list"]) == 0:
        kfold_dataset["types_list"] = None
        test_dataset["types_list"] = None

    return kfold_dataset, test_dataset


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
