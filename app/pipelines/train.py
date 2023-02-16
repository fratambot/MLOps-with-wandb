import argparse
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

# use "*/app" folder as base_path for local imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# local import
from models.models import CNN_model  # noqa: E402
from utils.data import datasets_loader  # noqa: E402


default_config = SimpleNamespace(
    # hyperparams
    batch_size=64,
    lr=0.01,
    epochs=5,
    dropout_1=0.2,
    dropout_2=0.2,
    dense=128,
    dropout_3=0.2,
    # tuning
    tune=False,
    max_sweep=30,
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
        help="retrain an existing model. It requires the "
        "WANDB_MODEL_RETRAIN env variable",
    )
    args = parser.parse_args()
    return args


def train(config=None):
    wandb.login()
    artifact_folder_path = os.path.join(base_path, "artifacts", "data")
    artifact_name = "split-data"
    artifact_extension = ".npz"
    artifact_filepath = os.path.join(
        artifact_folder_path, artifact_name + artifact_extension
    )
    print("data art filepath = ", artifact_filepath)
    datasets = datasets_loader(
        config.project, artifact_folder_path, artifact_name, artifact_filepath
    )
    if config.retrain:
        print("Retraining an existing model")
        with wandb.init(
            project=config.project,
            job_type="retraining",
            tags=["candidate"],
            config=config,
        ) as run:
            print("args passed = ", config)
            artifact_folder_path = os.path.join(base_path, "artifacts", "models")
            artifact_name = "CNN_model"
            artifact_extension = ".h5"
            artifact_filepath = os.path.join(
                artifact_folder_path, artifact_name + artifact_extension
            )
            wandb_artifact_path = os.path.join(
                config.entity, "model-registry", config.model_id
            )

            artifact = run.use_artifact(wandb_artifact_path, type="model")
            producer_run = artifact.logged_by()
            # update producer_run config with passed config
            producer_run.config.update(vars(config))
            # inject into wanb.config
            wandb.config.update(producer_run.config)
            final_config = wandb.config
            print("run config = ", final_config)
            artifact.download(artifact_folder_path)
            model = keras.models.load_model(artifact_filepath)
            model.fit(
                datasets.X_train,
                datasets.y_train,
                epochs=final_config.epochs,
                batch_size=final_config.batch_size,
                validation_data=(datasets.X_val, datasets.y_val),
                callbacks=[WandbMetricsLogger()],
            )
            model.save(artifact_filepath)
            retrained_model_artifact = wandb.Artifact(
                artifact_name,
                type="model",
                description="A simple CNN classifier for MNIST",
                metadata=vars(final_config),
            )
            retrained_model_artifact.add_file(local_path=artifact_filepath)
            run.log_artifact(retrained_model_artifact)

    else:
        with wandb.init(
            project=config.project, job_type=config.job_type, config=config
        ) as run:
            config = wandb.config
            print("run config = ", config)
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
            artifact_folder_name = "models"
            artifact_name = "CNN_model"
            artifact_extension = ".h5"
            artifact_filepath = os.path.join(
                base_path,
                "artifacts",
                artifact_folder_name,
                artifact_name + artifact_extension,
            )
            model.save(artifact_filepath)
            trained_model_artifact = wandb.Artifact(
                artifact_name,
                type="model",
                description="A simple CNN classifier for MNIST",
                metadata=vars(config),
            )
            trained_model_artifact.add_file(local_path=artifact_filepath)
            run.log_artifact(trained_model_artifact)

    return model


def tune(config):
    wandb.login()
    print("↑↑↑ Running sweeps in wandb...")
    with open(os.path.join(base_path, "pipelines", "sweep_config.yaml"), "r") as f:
        sweep_config = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_config, project=config.project)
    # functools.partial wizardry to pass a function with
    # default configs to wandb.agent using the sdk
    train_with_config = partial(train, config=config)
    wandb.agent(sweep_id=sweep_id, function=train_with_config, count=config.max_sweep)
    print("Stopping sweep agent from CLI...")
    run_id = os.path.join(config.entity, config.project, sweep_id)
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
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    if wandb_key is None or wandb_project is None or wandb_entity is None:
        print(
            "ERROR: You need to set the WANDB_API_KEY, WANDB_PROJECT and "
            "WANDB_ENTITY env variables to use this script"
        )
    else:
        env_config = SimpleNamespace(project=wandb_project, entity=wandb_entity)
        args = parse_args()
        if args.tune:
            print("=== Model tuning pipeline ===")
            # add env_config to default_config
            vars(default_config).update(vars(env_config))
            # override default_config with args
            vars(default_config).update(vars(args))
            default_config.job_type = "tuning"
            tune(default_config)
        elif args.retrain:
            model_to_retrain = os.environ.get("WANDB_MODEL_RETRAIN")
            if model_to_retrain is None:
                print(
                    "ERROR: you must set the id of the model to retrain in the"
                    "env variable, e.g. WANDB_MODEL_RETRAIN='CNN_MNIST:v0'"
                )
            else:
                # use only the passed args
                partial_args = only_passed_args(args)
                # add model_id to env_config and update passed_args
                env_config.model_id = model_to_retrain
                vars(partial_args).update(vars(env_config))
                partial_args.job_type = "retraining"
                train(partial_args)
        else:
            print("=== Model training pipeline ===")
            # add env_config to default_config
            vars(default_config).update(vars(env_config))
            # override default_config with args
            vars(default_config).update(vars(args))
            default_config.job_type = "training"
            train(default_config)
        print("=== Finished ===")
