import argparse
import json
import numpy as np
import os
import sys
import wandb

from dataclasses import dataclass
from dotenv import load_dotenv
from tensorflow.keras.utils import to_categorical
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


def train(args):
    config_filepath = os.path.join(base_path, "config.json")
    global CONFIG
    CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = "MNIST"
    print("↑↑↑ Loading datasets from wandb...")
    with wandb.init(project=PROJECT_NAME, job_type="load") as run:
        split_data_filename = CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
            "filename"
        ]
        split_data_root_dir = CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
            "root_dir"
        ]
        split_data_filepath = CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
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

    train_config = {"epochs": args.train_epochs, "batch_size": 64, "lr": 0.01}
    with wandb.init(
        project=PROJECT_NAME, job_type="training", config=train_config
    ) as run:
        config = wandb.config
        print("Building model")
        model = CNN_model(config.lr)

        model.fit(
            datasets.X_train,
            to_categorical(datasets.y_train),
            epochs=config.epochs,
            batch_size=config.batch_size,
            validation_data=(datasets.X_val, to_categorical(datasets.y_val)),
            callbacks=[WandbMetricsLogger()],
        )

        CONFIG[PROJECT_NAME]["artifacts"]["models"] = {"filename": "CNN_model_baseline"}
        model_filename = CONFIG[PROJECT_NAME]["artifacts"]["models"]["filename"]
        model_root_dir = os.path.join(base_path, "artifacts/models")
        CONFIG[PROJECT_NAME]["artifacts"]["models"]["root_dir"] = model_root_dir
        model_filepath = os.path.join(model_root_dir, model_filename + ".h5")
        CONFIG[PROJECT_NAME]["artifacts"]["models"]["filepath"] = model_filepath
        CONFIG[PROJECT_NAME]["artifacts"]["models"]["train_config"] = dict(train_config)
        model.save(model_filepath)
        trained_model_artifact = wandb.Artifact(
            model_filename,
            type="model",
            description="A simple CNN classifier for MNIST - baseline",
            metadata=dict(train_config),
        )
        trained_model_artifact.add_file(local_path=model_filepath)
        wandb.save(base_path=model_filepath)
        run.log_artifact(trained_model_artifact)

        # save updates to CONFIG file
        with open(config_filepath, "w") as f:
            json.dump(CONFIG, f, indent=2)

    return model


@dataclass
class DataSets:
    X_train: float
    y_train: float
    X_val: float
    y_val: float
    X_test: float
    y_test: float


if __name__ == "__main__":
    # python app/pipelines/train.py
    # Parse args
    docstring = """By default this script perform a simple training. You can tune the hyperparams if you want. See the help """  # noqa: E501
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train_epochs", default=10, type=int)
    args = parser.parse_args()

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("=== Model training pipeline ===")
        train(args)
        print("=== Finished ===")
