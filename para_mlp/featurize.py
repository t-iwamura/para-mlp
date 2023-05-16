from itertools import chain, product
from typing import List, Tuple

import mlpcpp
import numpy as np
from numpy.typing import NDArray
from pymatgen.core.structure import Structure

from para_mlp.data_structure import ModelParams
from para_mlp.preprocess import make_force_id


class RotationInvariant:
    """
    Class to calculate rotation invariants from model parameters and structure set
    """

    def __init__(
        self,
        model_params: ModelParams = None,
    ) -> None:

        self._model_params: ModelParams = model_params

    @property
    def model_params(self) -> ModelParams:
        """Return ModelParams class set to the instance

        Returns:
            ModelParams: The dataclass about model parameters
        """
        return self._model_params

    def __call__(
        self,
        structure_set: List[Structure],
        n_structure_list: List[int],
        types_list: List[List[int]] = None,
    ) -> NDArray:
        """Calculate feature matrix from given structures

        Args:
            structure_set (List[Structure]): structure set.
                List of pymatgen Structure class instances.
            n_structure_list (List[int]): list of n_structure for each sub dataset.
            types_list (List[List[int]], optional): list of element types
                about each structure. Defaults to None.

        Returns:
            NDArray: The feature matrix. The shape is as follows
                shape=({n_st_dataset}, ?)
            If use_force is True, a matrix whose shape is
                shape=(3 * {number of atoms in structure} * {n_st_dataset}, ?)
            is joined below energy feature matrix.
        """
        x = self.calculate_feature(structure_set, n_structure_list, types_list)

        return x

    def make_struct_params(
        self, structure_set: List[Structure], types_list: List[List[int]] = None
    ) -> Tuple[List[NDArray], List[NDArray], List[List[int]], List[int]]:
        """Make structure parameters

        Args:
            structure_set (List[Structure]): structure set
            types_list (List[List[int]], optional): list of element types
                about each structure. Defaults to None.

        Returns:
            Tuple[List[NDArray], List[NDArray], List[List[int]], List[int], List[int]]:
            In order
            lattice_matrix: The array of axis vectors of structures. This variable
                corresponds to 'axis_array' in mlptools.
            coords: The list of cartesian coordinates matrix. This variable corresponds
                to 'positions_c_array' in mlptools.
            types: The array of atom id. This variable corresponds to 'types_array'
                in mlptools. The id is allocated like 0, 1, ...
            atom_num_in_structure: The number of atoms in structures. This variable
                corresponds to 'n_atoms_all' in mlptools.
        """
        lattice_matrix = [
            structure.lattice.matrix.transpose() for structure in structure_set
        ]
        coords = [
            np.array([sites.coords for sites in structure.sites]).transpose()
            for structure in structure_set
        ]

        if types_list is not None:
            types = types_list
        elif self.model_params.composite_num == 2:
            up_moments = [0 for _ in range(16)]
            down_moments = [1 for _ in range(16)]
            all_moments = up_moments + down_moments
            types = [all_moments for _ in structure_set]
        else:
            types = [[0 for _ in structure.sites] for structure in structure_set]
        atom_num_in_structure = [len(structure.sites) for structure in structure_set]

        return (
            lattice_matrix,
            coords,
            types,
            atom_num_in_structure,
        )

    def calculate_feature(
        self,
        structure_set: List[Structure],
        n_structure_list: List[int],
        types_list: List[List[int]] = None,
    ) -> NDArray:
        """Calculate feature matrix

        Args:
            structure_set (List[Structure]): structure set.
                List of pymatgen Structure class.
            n_structure_list (List[int]): list of n_structure for each sub dataset.
            types_list (List[List[int]], optional): list of element types
                about each structure. Defaults to None.

        Returns:
            NDArray: feature matrix
        """
        # Make structure parameters
        (
            axis_array,
            positions_c_array,
            types_array,
            n_atoms_all,
        ) = self.make_struct_params(structure_set, types_list)

        # Make feature parameters
        feature_params = self.model_params.make_feature_params()

        # Make the other parameters
        n_sub_dataset = len(n_structure_list)
        use_force_list = [
            int(self.model_params.use_force) for _ in range(n_sub_dataset)
        ]

        _feature_object = mlpcpp.PotentialModel(
            axis_array,
            positions_c_array,
            types_array,
            self.model_params.composite_num,
            self.model_params.use_force,
            feature_params["radial_params"],
            self.model_params.cutoff_radius,
            self.model_params.radial_func,
            self.model_params.feature_type,
            self.model_params.polynomial_model,
            self.model_params.polynomial_max_order,
            self.model_params.lmax,
            feature_params["lm_seq"],
            feature_params["l_comb"],
            feature_params["lm_coeffs"],
            n_structure_list,
            use_force_list,
            n_atoms_all,
            False,
            self.model_params.is_paramagnetic,
            self.model_params.delta_learning,
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
                    feature_column_id = make_force_id(
                        str(sid + 1).zfill(5), center, 0, n_atom=32
                    )
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[0] - sites[neighbor].coords[0]
                    )
                    # Calculate y component
                    feature_column_id = make_force_id(
                        str(sid + 1).zfill(5), center, 1, n_atom=32
                    )
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[1] - sites[neighbor].coords[1]
                    )
                    # Calculate z component
                    feature_column_id = make_force_id(
                        str(sid + 1).zfill(5), center, 2, n_atom=32
                    )
                    force_feature[
                        feature_column_id, coeff_orders_id
                    ] += force_common * (
                        sites[center].coords[2] - sites[neighbor].coords[2]
                    )
            feature_matrix = np.concatenate((energy_feature, force_feature), axis=0)
        else:
            feature_matrix = energy_feature

        return feature_matrix
