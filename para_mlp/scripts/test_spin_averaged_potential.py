import logging
from pathlib import Path

import click
from lammps_api.pred import predict_energy_and_force

from para_mlp.pred import calc_spin_average, predict_property


@click.command()
@click.option("--pair_potential", required=True, help="path to pair potential.")
@click.option("--gtinv_potential", required=True, help="path to gtinv potential.")
@click.option("--averaged_potential", required=True, help="path to averaged potential.")
def main(pair_potential: str, gtinv_potential: str, averaged_potential: str) -> None:
    logging.basicConfig(level=logging.INFO)

    nano_cluster_dir_path = Path(averaged_potential).parent / "nano_cluster"
    if not nano_cluster_dir_path.exists():
        nano_cluster_dir_path.mkdir()

    lammps_structures_dir_path = (
        Path.home()
        / "lammps_api"
        / "data"
        / "inputs"
        / "paramagnetic_Fe"
        / "structures"
    )
    nano_cluster_lammps_path = (
        lammps_structures_dir_path / "nano_cluster" / "lammps_structure"
    )
    averaged_energy_dict = predict_energy_and_force(
        potential_file=averaged_potential,
        structure_file=str(nano_cluster_lammps_path),
        outputs_dir=str(nano_cluster_dir_path),
    )

    logging.info(" End LAMMPS calculation")

    nano_cluster_json_path = (
        Path.home() / "para-mlp" / "data" / "inputs" / "structure" / "nano_cluster.json"
    )
    pair_potential_path = Path(pair_potential)
    spin_average_dict = calc_spin_average(
        model_dir=str(pair_potential_path.parent),
        structure_file=str(nano_cluster_json_path),
    )

    gtinv_potential_path = Path(gtinv_potential)
    types_list = [0 for _ in range(9)]
    gtinv_energy_dict = predict_property(
        model_dir=str(gtinv_potential_path.parent),
        structure_file=str(nano_cluster_json_path),
        types_list=[types_list],
        use_force=False,
    )

    total_energy_sum = spin_average_dict["energy"] + gtinv_energy_dict["energy"]
    expected_energy = averaged_energy_dict["energy"]

    logging.info(f" spin averaged energy + gtinv energy: {total_energy_sum}")
    logging.info(f" Expected energy: {expected_energy}")


if __name__ == "__main__":
    main()
