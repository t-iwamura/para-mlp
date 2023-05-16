import json
from pathlib import Path

import click

from para_mlp.config import Config


@click.command()
@click.argument("data_dir_list", nargs=-1)
@click.option(
    "--composite_num",
    type=int,
    default=2,
    show_default=True,
    help="The number of composite elements.",
)
@click.option(
    "--polynomial_model",
    type=int,
    default=3,
    show_default=True,
    help="The ID of polynomial potential model.",
)
@click.option(
    "--polynomial_max_order",
    type=int,
    default=3,
    show_default=True,
    help="The maximum of polynomial order.",
)
@click.option(
    "--cutoff",
    type=float,
    default=6.0,
    show_default=True,
    help="The cutoff radius in neighboring atomic density.",
)
@click.option(
    "--n_radial_function",
    type=int,
    default=20,
    show_default=True,
    help="The number of radial functions.",
)
@click.option(
    "--gtinv_lmax",
    default="4,2",
    show_default=True,
    help="The maximums of azimuthal quantum numbers for rotational invariants.",
)
@click.option(
    "--alpha",
    default=1e-3,
    show_default=True,
    help="The regularization parameter of Ridge regression.",
)
@click.option("--model_dir", required=True, help="Path to model directory.")
def main(
    data_dir_list,
    composite_num,
    polynomial_model,
    polynomial_max_order,
    cutoff,
    n_radial_function,
    gtinv_lmax,
    alpha,
    model_dir,
) -> None:
    """User interface to arrange model.json for machine learning potential estimation"""
    config = Config()
    config.data_dir_list = tuple(data_dir_list)
    config.composite_num = composite_num
    config.polynomial_model = polynomial_model
    config.polynomial_max_order = polynomial_max_order
    config.is_paramagnetic = True
    config.cutoff_radius_min = cutoff
    config.cutoff_radius_max = cutoff
    config.gaussian_params2_flag = 2
    config.gaussian_params2_num_min = n_radial_function
    config.gaussian_params2_num_max = n_radial_function
    config.gtinv_lmax = tuple(int(lmax) for lmax in gtinv_lmax.split(","))
    config.use_force = True
    config.alpha = (alpha,)
    config.save_log = True
    config.model_dir = model_dir

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        model_dir_path.mkdir(parents=True)

    json_path = model_dir_path / "model.json"
    with json_path.open("w") as f:
        json.dump(config.to_dict(), f, indent=4)  # type: ignore


if __name__ == "__main__":
    main()
