import os
import sys
from tempfile import NamedTemporaryFile

from mlp_build_tools.mlpgen.myIO import ReadVaspruns
from pymatgen.core.structure import Structure
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/mlp_build_tools/cpp/lib")


def make_vasprun_tempfile(structure_ids: tuple = None):
    if structure_ids is None:
        raise TypeError("Receive NoneType object.")

    inputs_dir = os.path.dirname(os.path.abspath(__file__)) + "/../data/inputs/data/"
    temp_object = NamedTemporaryFile(mode="w", delete=False)
    for sid in structure_ids:
        print(inputs_dir + sid + "/vasprun.xml_1_type", file=temp_object)
    temp_object.close()

    return temp_object.name


def create_dataset(structure_ids: tuple = None):
    if structure_ids is None:
        structure_ids = (str(i + 1).zfill(5) for i in range(5000))

    vasprun_tempfile = make_vasprun_tempfile(structure_ids)

    energy, force, stress, seko_structures, volume = ReadVaspruns(vasprun_tempfile).get_data()

    structures = [
        Structure(struct.get_axis().transpose(), struct.get_elements(), struct.get_positions().transpose())
        for struct in seko_structures
    ]

    dataset = {"energy": energy, "structures": structures}

    return dataset


def split_dataset(dataset: dict = None, test_size: float = 0.1):
    structure_train, structure_test, y_train, y_test = train_test_split(
        dataset["structures"], dataset["energy"], test_size=test_size, shuffle=True
    )

    return structure_train, structure_test, y_train, y_test
