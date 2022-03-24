import os
import sys
from typing import List

import numpy as np

from para_mlp.data_structure import ModelParams

sys.path.append((os.path.dirname(os.path.abspath(__file__))) + "mlp_build_tools/cpp/lib")


class RotationInvariant:
    def __init__(
        self,
        structure_set: List = None,
        model_params: ModelParams = None,
    ):
        self._lattice_matrix = [structure.lattice.matrix.transpose() for structure in structure_set]
        self._coords = [np.array([sites.coords for sites in structure.sites]).transpose() for structure in structure_set]
        self._types = [[0 for site in structure.sites] for structure in structure_set]
        self._length_of_structures = [len(structure_set)]
        self._atom_num_in_structure = [len(structure.sites) for structure in structure_set]

        self._model_params = model_params
        self._x = None

    @property
    def axis_array(self):
        return self._lattice_matrix

    @property
    def positions_c_array(self):
        return self._coords

    @property
    def types_array(self):
        return self._types

    @property
    def n_atoms_all(self):
        return self._atom_num_in_structure

    @property
    def n_st_dataset(self):
        return self._length_of_structures

    @property
    def x(self):
        if self._x is None:
            import mlpcpp

            _feature_object = mlpcpp.PotentialModel(
                self.axis_array,
                self.positions_c_array,
                self.types_array,
                self._model_params.composite_num,
                self._model_params.use_force,
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
                [0],
                self.n_atoms_all,
                False,
            )
            self._x = _feature_object.get_x()

        return self._x
