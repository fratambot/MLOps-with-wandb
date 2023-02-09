import argparse
import json
import numpy as np
import os
import sys
import wandb
import yaml

from dotenv import load_dotenv
from functools import partial
from tensorflow.keras.utils import to_categorical
from types import SimpleNamespace
from wandb.keras import WandbMetricsLogger

# load env variables
load_dotenv()

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local import
from models.models import CNN_model  # noqa: E402
from utils.dataclass import DataSets  # noqa: E402

default_config = SimpleNamespace(
    batch_size=128,
    lr=0.01,
    epochs=5,
    dropout_1=0.4,
    dropout_2=0.4,
    dense=128,
    dropout_3=0.4,
    tune=False,
    max_sweep=20,
)


def parse_args():
    docstring = """Overriding default argments"""
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--batch_size", type=int, default=default_config.batch_size, help="batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=default_config.epochs,
        help="number of training epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=default_config.lr, help="learning rate"
    )
    parser.add_argument(
        "--dropout_1",
        type=float,
        default=default_config.dropout_1,
        help="1st dropout layer",
    )
    parser.add_argument(
        "--dropout_2",
        type=float,
        default=default_config.dropout_2,
        help="2nd dropout layer",
    )
    parser.add_argument(
        "--dense",
        type=float,
        default=default_config.dense,
        help="units for dense layer",
    )
    parser.add_argument(
        "--dropout_3",
        type=float,
        default=default_config.dropout_3,
        help="3rd dropout layer",
    )
    parser.add_argument(
        "--tune",
        default=default_config.tune,
        action=argparse.BooleanOptionalAction,
        help="run hyperparameters tuning on wandb",
    )
    parser.add_argument(
        "--max_sweep",
        type=int,
        default=default_config.max_sweep,
        help="maximum number of sweeps (=count for wandb.agent)",
    )
    args = parser.parse_args()
    return args


def train(config=None):
    config_filepath = os.path.join(base_path, "config.json")
    global LOCAL_CONFIG
    LOCAL_CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = "MNIST"
    print("↑↑↑ Loading datasets from wandb...")
    with wandb.init(project=PROJECT_NAME, job_type="load") as run:
        split_data_filename = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
            "filename"
        ]
        split_data_root_dir = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
            "root_dir"
        ]
        split_data_filepath = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
            "filepath"
        ]
        run.use_artifact(split_data_filename + ":latest").download(split_data_root_dir)
        datasets = np.load(split_data_filepath, allow_pickle=True)
        # Unzip datasets and rebuild dataclass
        datasets = DataSets(
            datasets["X_train"],
            datasets["y_train"],
            datasets["X_val"],
            datasets["y_val"],
            datasets["X_test"],
            datasets["y_test"],
        )

        print("training examples: ", len(datasets.X_train))
        print("validation examples: ", len(datasets.X_val))

    with wandb.init(project=PROJECT_NAME, job_type="training", config=config) as run:
        # good practice to inject params using sweep
        config = wandb.config
        print("Building model")
        model = CNN_model(config)

        model.fit(
            datasets.X_train,
            to_categorical(datasets.y_train),
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(datasets.X_val, to_categorical(datasets.y_val)),
            callbacks=[WandbMetricsLogger()],
        )

        LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"] = {"filename": "CNN_model"}
        model_filename = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"]["filename"]
        model_root_dir = os.path.join(base_path, "artifacts/models")
        LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"]["root_dir"] = model_root_dir
        model_filepath = os.path.join(model_root_dir, model_filename + ".h5")
        LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"]["filepath"] = model_filepath
        model.save(model_filepath)
        trained_model_artifact = wandb.Artifact(
            model_filename,
            type="model",
            description="A simple CNN classifier for MNIST",
            metadata=vars(config),
        )
        trained_model_artifact.add_file(local_path=model_filepath)
        wandb.save(base_path=model_filepath)
        run.log_artifact(trained_model_artifact)

    # save updates to LOCAL_CONFIG file
    with open(config_filepath, "w") as f:
        json.dump(LOCAL_CONFIG, f, indent=2)

    return model


def tune(config):
    print("config in tune = ", config)
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = "MNIST"
    print("↑↑↑ Running sweeps in wandb...")
    with open(os.path.join(base_path, "artifacts/models", "sweep.yaml"), "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)
    train_with_config = partial(train, config=config)
    wandb.agent(sweep_id=sweep_id, function=train_with_config, count=config.max_sweep)

    return


if __name__ == "__main__":
    # python app/pipelines/train.py --tune

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        args = parse_args()
        vars(default_config).update(vars(args))
        print("=== Model training pipeline ===")
        if args.tune:
            print("Running sweeps on wandb")
            tune(default_config)
        else:
            # Override default_config with args
            train(default_config)
        print("=== Finished ===")
