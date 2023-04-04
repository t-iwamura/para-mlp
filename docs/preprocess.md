# Preprocess

## Dataset Generation

1. Relax the original structure.

2. Generate a bunch of structures from the relaxed structure.

    Run the following jupyter notebook, `~/para-mlp/exp/notebooks/generate_structure_set`.

3. Calculate the energies and atomic forces for the generated structures.

## Dataset Processing

1. Rsync necessary data to training environment.
- `POSCAR`
- `vasprun.xml`
- `input` directory

2. Create data for easy dataset loading.
- `targets.json`
- `types_list.json`
- `structure.json`
- `energy.npy` and `force.npy`

3. Generate the setting files to split dataset.
- `structure_id`
- `yids_for_kfold`
- `yids_for_test`

4. Delete `vasprun.xml`.
