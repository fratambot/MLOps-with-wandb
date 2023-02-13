import argparse
import json
import os
import subprocess
import sys
import wandb
import yaml

from dotenv import load_dotenv
from functools import partial
from tensorflow import keras
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
from utils.data import datasets_loader  # noqa: E402

default_config = SimpleNamespace(
    # hyperparams
    batch_size=32,
    lr=0.01,
    epochs=5,
    dropout_1=0.4,
    dropout_2=0.4,
    dense=128,
    dropout_3=0.4,
    # tuning
    tune=False,
    max_sweep=3,
    # retraining
    retrain=False,
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
    parser.add_argument(
        "--retrain",
        default=default_config.retrain,
        action=argparse.BooleanOptionalAction,
        help="run hyperparameters tuning on wandb",
    )
    args = parser.parse_args()
    return args


def train(config=None):
    config_filepath = os.path.join(base_path, "config.json")
    global LOCAL_CONFIG
    LOCAL_CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = os.environ.get("WANDB_PROJECT")
    datasets = datasets_loader(LOCAL_CONFIG, PROJECT_NAME)
    if config.retrain:
        print("Retraining an existing model")
        with wandb.init(
            project=PROJECT_NAME,
            job_type="retraining",
            tags=["candidate"],
            config=config,
        ) as run:
            print("my default config in wandb = ", config)
            model_root_dir = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "root_dir"
            ]
            model_filepath = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "filepath"
            ]
            artifact_path = os.path.join(
                os.environ.get("WANDB_ENTITY"), os.environ.get("WANDB_MODEL_RETRAIN")
            )
            artifact = run.use_artifact(artifact_path, type="model")
            producer_run = artifact.logged_by()
            producer_run.config.update(vars(config))
            wandb.config.update(producer_run.config)
            final_config = wandb.config
            print("final_config = ", final_config)
            artifact.download(model_root_dir)
            model = keras.models.load_model(model_filepath)
            print("artifact model successfully loaded")
            model.fit(
                datasets.X_train,
                datasets.y_train,
                epochs=final_config.epochs,
                batch_size=final_config.batch_size,
                validation_data=(datasets.X_val, datasets.y_val),
                callbacks=[WandbMetricsLogger()],
            )
            # This overwrites the local model if present
            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"] = {
                "filename": "CNN_model"
            }
            model_filename = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "filename"
            ]
            model_root_dir = os.path.join(base_path, "artifacts/models")
            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "root_dir"
            ] = model_root_dir
            model_filepath = os.path.join(model_root_dir, model_filename + ".h5")
            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "filepath"
            ] = model_filepath
            model.save(model_filepath)
            retrained_model_artifact = wandb.Artifact(
                model_filename,
                type="model",
                description="A simple CNN classifier for MNIST",
                metadata=vars(config),
            )
            retrained_model_artifact.add_file(local_path=model_filepath)
            wandb.save(base_path=model_filepath)
            run.log_artifact(retrained_model_artifact)

    else:
        with wandb.init(
            project=PROJECT_NAME, job_type=config.job_type, config=config
        ) as run:
            # good practice to inject params using sweep
            config = wandb.config
            print("Building model")
            model = CNN_model(config)

            model.fit(
                datasets.X_train,
                datasets.y_train,
                epochs=config.epochs,
                batch_size=config.batch_size,
                validation_data=(datasets.X_val, datasets.y_val),
                callbacks=[WandbMetricsLogger()],
            )

            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"] = {
                "filename": "CNN_model"
            }
            model_filename = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "filename"
            ]
            model_root_dir = os.path.join(base_path, "artifacts/models")
            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "root_dir"
            ] = model_root_dir
            model_filepath = os.path.join(model_root_dir, model_filename + ".h5")
            LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"][
                "filepath"
            ] = model_filepath
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
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = "MNIST"
    print("↑↑↑ Running sweeps in wandb...")
    with open(os.path.join(base_path, "pipelines", "sweep_config.yaml"), "r") as f:
        sweep_config = yaml.safe_load(f)
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)
    # partial wizardry to pass a function with default configs to wandb.agent
    train_with_config = partial(train, config=config)
    wandb.agent(sweep_id=sweep_id, function=train_with_config, count=config.max_sweep)
    print("Stopping sweep agent from CLI...")
    entity = os.environ.get("WANDB_ENTITY")
    run_id = os.path.join(entity, PROJECT_NAME, sweep_id)
    response = subprocess.run(
        ["wandb", "sweep", "--stop", run_id], stderr=subprocess.PIPE, text=True
    )
    print(response)
    return


def only_passed_args(args):
    partial_args_set = set(vars(args).items()) - set(vars(default_config).items())
    partial_args_dict = {}
    for tuples in partial_args_set:
        partial_args_dict[tuples[0]] = tuples[1]
    partial_args = argparse.Namespace(**partial_args_dict)

    return partial_args


if __name__ == "__main__":
    # python app/pipelines/train.py
    # python app/pipelines/train.py --tune
    # python app/pipelines/train.py --retrain --epochs=10

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        args = parse_args()
        if args.tune:
            print("=== Model tuning pipeline ===")
            # Override default_config with args
            vars(default_config).update(vars(args))
            default_config.job_type = "tuning"
            tune(default_config)
        elif args.retrain:
            # TODO: add model retrain env
            # Use only the passed args
            partial_args = only_passed_args(args)
            partial_args.job_type = "retraining"
            train(partial_args)
        else:
            print("=== Model training pipeline ===")
            # Override default_config with args
            vars(default_config).update(vars(args))
            default_config.job_type = "training"
            train(default_config)
        print("=== Finished ===")
