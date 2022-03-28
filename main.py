#!/usr/bin/env python
from para_mlp.config import Config
from para_mlp.preprocess import create_dataset, split_dataset
from para_mlp.train import dump_model, train_and_eval


def main(config: Config):
    dataset = create_dataset(config.data_dir, config.targets_json)
    kfold_dataset, test_dataset = split_dataset(dataset, shuffle=config.shuffle)

    best_model, best_model_params = train_and_eval(config, kfold_dataset, test_dataset)

    dump_model(best_model, best_model_params, config.model_dir)


if __name__ == "__main__":
    config = Config()
    main(config)
