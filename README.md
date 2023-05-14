# para-mlp

Python package to create paramagnetic machine learning potentials

## Overview

This package creates machine learning potentials for paramagnetic systems. It generate structures for dataset and perform preprocessing for machine learning. By using a dataset, it creates machine learning potentials by machine learning.
To use this package, you have to install Python packages, `mlp_build_tools` and  `mlpcpp`, maintained by the author of this repository.

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
  --data_dir TEXT             the path to data directory.  [required]
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

# See generated files
$ ls
data/ relax/ structure_generation.log
$ ls data
00001/ 00002/ 00003/ ...
```
3. Run VASP calculations for the generated structures and get total energy and atomic forces for the each structure.

### Preprocessing

To make dataset loading easy, you have to do preprocessing. Suppose that you made 5000 structures in a dataset directory `para-mlp/data/fcc` and performed VASP calculations for each structure. Run an following command.
```shell
$ para-mlp process --data_dir para-mlp/data/fcc --structure_id_max 5000

# See generated files
$ ls processing
energy.npy force.npy targets.json types_list.json use_force_too/
$ ls data/00001
structure.json ...
```

### Training

Now that you've finished preprocessing, you can create machine learning potentials by training.
First, create model directory `para-mlp/models/fcc/001`.
Next, arrange a json file in the directory, which configures parameters of a machine learning potential. You can use a Python program as shown below.
```shell
# Display help
$ python <para-mlp root>/scripts/arrange_model_json.py --help

# Arrange model.json
$ python <para-mlps root>/scripts/arrange_model_json.py --model_dir para-mlp/models/fcc/001
```

Now, you've completed all the necessary steps for machine learning potential estimation ! By a following command, you can perform machine learning and create an machine learning potential.

```shell
$ para-mlp train para-mlp/models/fcc/001/model.json
```
