# para-mlp

Python package to create paramagnetic machine learning potentials

## Overview

## Installation

```shell
$ cd <para-mlp's root>
$ pip install .
```

## Usage

You can display helpfull messages by executing a following command.

```shell
$ para-mlp --help
Usage: para-mlp [OPTIONS] COMMAND [ARGS]...

  Python package to create paramagnetic machine learning potentials

Options:
  --help  Show this message and exit.

Commands:
  generate  Generate deformed structures for dataset generation
  pareto    search pareto optimal potentials
  predict   predict energy and force by machine learning potential
  process   Process raw dataset for easy dataset loading
  train     train machine learning potential
```

Also, you can display the details of each subcommand.

```shell
$ para-mlp process --help
Usage: para-mlp process [OPTIONS]

  Process raw dataset for easy dataset loading

Options:
  --data_dir_name TEXT        the name of data directory.  [required]
  --structure_id_max INTEGER  the maximum of structure ID.  [required]
  --addition / --no-addition
  --old_data_dir_name TEXT    the name of old data directory.
  --atomic_energy FLOAT       the energy of isolated Fe atom.  [default:
                              -3.37689]
  --help                      Show this message and exit.
```


### Structure Set Generation

1. Suppose that you use FCC as a prototype structure and create a dataset directory in `para-mlp/data/fcc`. Relax the prototype structure. Then, place the relaxed structure, `POSCAR` to `para-mlp/data/fcc/relax`.
2. Run following commands, and you'll generate 5000 deformed structures. The structure format of the structures is POSCAR.
```shell
$ cd para-mlp/data/fcc
$ para-mlp generate --root_dir . --n_structure 5000 2> structure_generation.log

# See the generated files
$ ls
data/ relax/ structure_generation.log
$ ls data
00001/ 00002/ 00003/ ...
```
3. Run VASP calculations for the generated structures and get total energy and atomic forces for the each structure.

### Preprocessing
