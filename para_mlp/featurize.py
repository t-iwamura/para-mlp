import sys
from itertools import chain, product
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure

from para_mlp.data_structure import ModelParams
from para_mlp.preprocess import make_force_id

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

    def __call__(self, structure_set: List[Structure]) -> NDArray:
        """Calculate feature matrix from given structures

        Args:
            structure_set (List[Structure]): structure set.
                List of pymatgen Structure class instances.

        Returns:
            NDArray: feature matrix
        """
        self.set_struct_params(structure_set)
        x = self.calculate_feature()

        return x

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

    def calculate_feature(self) -> NDArray:
        """Calculate feature matrix

        Returns:
            NDArray: The feature matrix. The shape is as follows
                    shape=({n_st_dataset}, ?)
                If use_force is True, a matrix whose shape is
                    shape=(3 * {number of atoms in structure} * {n_st_dataset}, ?)
                is joined below energy feature matrix.
        """
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
        x = _feature_object.get_x()

        if self._model_params.use_force:
            fbegin, sbegin = (
                _feature_object.get_fbegin()[0],
                _feature_object.get_sbegin()[0],
            )
            feature_ids = [
                fid
                for fid in chain.from_iterable(
                    [range(sbegin), range(fbegin, x.shape[0])]
                )
            ]
            return x[feature_ids]
        else:
            return x


class SpinFeaturizer:
    """
    Class to calculate Heisenberg model type spin feature from structure set
    """

    def __init__(self, model_params: ModelParams) -> None:
        self._magnetic_cutoff_radius = model_params.magnetic_cutoff_radius
        self._coeff_order_max = model_params.coeff_order_max
        self._coeff_orders = [i for i in range(2, self._coeff_order_max + 1)]

        self._use_force = model_params.use_force

    def __call__(self, structure_set: List[Structure]) -> NDArray:
        """Calculate Heisenberg model type feature matrix from given structure set

        Args:
            structure_set (List[Structure]): structure set

        Returns:
            NDArray: Feature matrix. The shape is as follows
                    shape=(n_struct_set, n_coeff_orders)
                If self._use_force is True, a matrix whose shape is
                    shape=(column_length, n_coeff_orders)
                is joined below energy feature matrix. The keyword "column_length" is
                    3 * {number of atoms in structure} * {n_struct_set}
                The keyword "n_coeff_orders" is the number of order of
                exchange interaction's constant.
        """
        magmom_upper = [1] * 16
        magmom_lower = [-1] * 16
        magmoms = [*magmom_upper, *magmom_lower]

        n_struct_set = len(structure_set)
        n_coeff_orders = len(self._coeff_orders)

        energy_feature = np.zeros((n_struct_set, n_coeff_orders))
        for sid, coeff_orders_id in product(range(n_struct_set), range(n_coeff_orders)):
            neighbors = structure_set[sid].get_neighbor_list(
                self._magnetic_cutoff_radius
            )
            for center, neighbor, _, distance in zip(*neighbors):
                energy_feature[sid, coeff_orders_id] += (
                    1
                    / (distance ** self._coeff_orders[coeff_orders_id])
                    * magmoms[center]
                    * magmoms[neighbor]
                )

        if self._use_force:
            feature_column_length = 3 * n_struct_set * len(structure_set[0].sites)
            force_feature = np.zeros((feature_column_length, n_coeff_orders))
            for sid, coeff_orders_id in product(
                range(n_struct_set), range(n_coeff_orders)
            ):
                neighbors = structure_set[sid].get_neighbor_list(
                    self._magnetic_cutoff_radius
                )
                sites = structure_set[sid].sites
                for center, neighbor, _, distance in zip(*neighbors):
                    coeff_order = self._coeff_orders[coeff_orders_id]
                    force_common = (
                        2
                        * coeff_order
                        / (distance ** (coeff_order + 2))
                        * magmoms[center]
                        * magmoms[neighbor]
                    )

                    # Calculate x component
                    feature_column_id = make_force_id(str(sid + 1).zfill(5), center, 0)
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[0] - sites[neighbor].coords[0]
                    )
                    # Calculate y component
                    feature_column_id = make_force_id(str(sid + 1).zfill(5), center, 1)
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[1] - sites[neighbor].coords[1]
                    )
                    # Calculate z component
                    feature_column_id = make_force_id(str(sid + 1).zfill(5), center, 2)
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[2] - sites[neighbor].coords[2]
                    )
            feature_matrix = np.concatenate((energy_feature, force_feature), axis=0)
        else:
            feature_matrix = energy_feature

        return feature_matrix
