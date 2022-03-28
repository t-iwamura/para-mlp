import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Tuple

from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split

mlp_build_tools_path = (
    Path.home() / "mlp-Fe" / "mlptools" / "mlp_build_tools" / "cpp" / "lib"
)
sys.path.append(mlp_build_tools_path.as_posix())


def make_vasprun_tempfile(
    structure_ids: Tuple[str, ...] = None, data_dir: str = "data"
) -> str:
    if structure_ids is None:
        raise TypeError("Receive NoneType object.")

    inputs_dir = Path(__file__).resolve().parent / ".." / data_dir / "inputs" / "data"

    temp_object = NamedTemporaryFile(mode="w", delete=False)
    for sid in structure_ids:
        vasprun_path = inputs_dir / sid / "vasprun.xml_1_type"
        print(vasprun_path.as_posix(), file=temp_object)
    temp_object.close()

    return temp_object.name


def create_dataset(
    structure_ids: Tuple[str, ...] = None, data_dir: str = "data"
) -> Dict[str, Any]:
    if structure_ids is None:
        structure_ids = tuple((str(i + 1).zfill(5) for i in range(5000)))

    vasprun_tempfile = make_vasprun_tempfile(structure_ids, data_dir=data_dir)

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


def split_dataset(
    dataset: Dict[str, Any] = None, test_size: float = 0.1, shuffle: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    structure_train, structure_test, y_train, y_test = train_test_split(
        dataset["structures"], dataset["energy"], test_size=test_size, shuffle=shuffle
    )

    kfold_dataset = {"structures": structure_train, "energy": y_train}
    test_dataset = {"structures": structure_test, "energy": y_test}

    return kfold_dataset, test_dataset
