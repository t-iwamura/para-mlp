import sys
from itertools import chain, product
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
    """
    Class to calculate rotation invariants from model parameters and structure set
    """

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
        """Calculate feature matrix from given structures

        Args:
            structure_set (List[Structure]): structure set.
                List of pymatgen Structure class instances.

        Returns:
            NDArray: feature matrix
        """
        self.set_struct_params(structure_set)
        self.calculate()

        return self._x

    def set_struct_params(self, structure_set: List[Structure]) -> None:
        """Set structure parameters

        The properties axis_array, positions_c_array, types_array,
        n_atoms_all, and n_st_dataset will be set.

        Args:
            structure_set (List[Structure]): structure set
        """
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
        """Return the array of axis vectors of structrures

        Returns:
            List[NDArray]: array of axis vectors of structures
        """
        return self._lattice_matrix

    @property
    def positions_c_array(self) -> List[NDArray]:
        """Return list of cartesian coordinates matrix

        Returns:
            List[NDArray]: List of matrix where cartesian coordinates of atoms
                are aligned
        """
        return self._coords

    @property
    def types_array(self) -> List[List[int]]:
        """Return array of atom id. The id is allocated like 0, 1, ...

        Returns:
            List[List[int]]: array of atom id
        """
        return self._types

    @property
    def n_atoms_all(self) -> List[int]:
        """Return number of atoms in structures

        Returns:
            List[int]: list of number of atoms in structures
        """
        return self._atom_num_in_structure

    @property
    def n_st_dataset(self) -> List[int]:
        """Return the length of structure list

        Returns:
            List[int]: the length of structure list
        """
        return self._length_of_structures

    @property
    def x(self) -> NDArray:
        """Return feature matrix

        Returns:
            NDArray: Feature matrix. The shape is as follows
                    shape=({n_st_dataset}, ?)
                If use_force is True, a matrix whose shape is
                    shape=(3 * {number of atoms in structure} * {n_st_dataset}, ?)
                is joined below energy feature matrix.
        """
        if self._x is None:
            self.calculate()

        return self._x

    def calculate(self) -> None:
        """Calculate feature matrix from given parameters"""
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


class SpinFeaturizer:
    """
    Class to calculate Heisenberg model type spin feature from structure set
    """

    def __init__(self, model_params: ModelParams) -> None:
        self._magnetic_cutoff_radius = model_params.magnetic_cutoff_radius
        self._coeff_order_max = model_params.coeff_order_max
        self._coeff_orders = [i for i in range(2, self._coeff_order_max + 1)]

    def __call__(self, structure_set: List[Structure]) -> NDArray:
        """Calculate Heisenberg model type feature matrix from given structure set

        Args:
            structure_set (List[Structure]): structure set

        Returns:
            NDArray: Feature matrix. The shape is as follows
                    shape=(len(structure_set), len(coeff_orders))
                The keyword 'coeff orders' refers to self._coeff_orders. This is
                the order of exchange interaction's constant.
        """
        magmom_upper = [1] * 16
        magmom_lower = [-1] * 16
        magmoms = [*magmom_upper, *magmom_lower]

        n_struct_set = len(structure_set)
        n_coeff_orders = len(self._coeff_orders)

        feature_matrix = np.zeros((n_struct_set, n_coeff_orders))
        for sid, coeff_orders_id in product(range(n_struct_set), range(n_coeff_orders)):
            neighbors = structure_set[sid].get_neighbor_list(
                self._magnetic_cutoff_radius
            )
            for center, neighbor, _, distance in zip(*neighbors):
                feature_matrix[sid, coeff_orders_id] += (
                    1
                    / (distance ** self._coeff_orders[coeff_orders_id])
                    * magmoms[center]
                    * magmoms[neighbor]
                )

        return feature_matrix
