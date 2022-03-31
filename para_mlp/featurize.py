import sys
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure

from para_mlp.data_structure import ModelParams

mlp_build_tools_path = (
    Path.home() / "mlp-Fe" / "mlptools" / "mlp_build_tools" / "cpp" / "lib"
)
sys.path.append(mlp_build_tools_path.as_posix())


class RotationInvariant:
    def __init__(
        self,
        model_params: ModelParams = None,
    ) -> None:

        self._model_params: ModelParams = model_params

        # Initialize structure parameters
        self._lattice_matrix: List[NDArray] = None
        self._coords: List[NDArray] = None
        self._types: List[List[int]] = None
        self._atom_num_in_structure: List[int] = None
        self._length_of_structures: List[int] = None

        # Initialize feature matrix
        self._x: NDArray = None

    def __call__(self, structure_set: List[Structure]) -> NDArray:
        self.set_struct_params(structure_set)
        self.calculate()

        return self._x

    def set_struct_params(self, structure_set: List[Structure]) -> None:
        self._lattice_matrix = [
            structure.lattice.matrix.transpose() for structure in structure_set
        ]
        self._coords = [
            np.array([sites.coords for sites in structure.sites]).transpose()
            for structure in structure_set
        ]
        self._types = [[0 for site in structure.sites] for structure in structure_set]
        self._length_of_structures = [len(structure_set)]
        self._atom_num_in_structure = [
            len(structure.sites) for structure in structure_set
        ]

    @property
    def axis_array(self) -> List[NDArray]:
        return self._lattice_matrix

    @property
    def positions_c_array(self) -> List[NDArray]:
        return self._coords

    @property
    def types_array(self) -> List[List[int]]:
        return self._types

    @property
    def n_atoms_all(self) -> List[int]:
        return self._atom_num_in_structure

    @property
    def n_st_dataset(self) -> List[int]:
        return self._length_of_structures

    @property
    def x(self) -> NDArray:
        if self._x is None:
            self.calculate()

        return self._x

    def calculate(self) -> None:
        import mlpcpp  # type: ignore

        _feature_object = mlpcpp.PotentialModel(
            self.axis_array,
            self.positions_c_array,
            self.types_array,
            self._model_params.composite_num,
            False,
            self._model_params.radial_params,
            self._model_params.cutoff_radius,
            self._model_params.radial_func,
            self._model_params.feature_type,
            self._model_params.polynomial_model,
            self._model_params.polynomial_max_order,
            self._model_params.lmax,
            self._model_params.lm_seq,
            self._model_params.l_comb,
            self._model_params.lm_coeffs,
            self.n_st_dataset,
            [int(self._model_params.use_force)],
            self.n_atoms_all,
            False,
        )
        _x = _feature_object.get_x()

        if self._model_params.use_force:
            fbegin, sbegin = (
                _feature_object.get_fbegin()[0],
                _feature_object.get_sbegin()[0],
            )
            feature_ids = [
                fid
                for fid in chain.from_iterable(
                    [range(sbegin), range(fbegin, _x.shape[0])]
                )
            ]
            self._x = _x[feature_ids]
        else:
            self._x = _x
