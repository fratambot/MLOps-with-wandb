import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import wandb

from dataclasses import dataclass, is_dataclass
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

# load env variables
load_dotenv()

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)


def data_preparation_pipeline(args):
    config_filepath = os.path.join(base_path, "config.json")
    global CONFIG
    CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = "MNIST"
    if args.build_datasets:
        with wandb.init(project=PROJECT_NAME, job_type="upload") as run:
            CONFIG[PROJECT_NAME] = {
                "artifacts": {"split-data": {"filename": "split-data"}}
            }
            split_data_filename = CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
                "filename"
            ]
            split_data_root_dir = os.path.join(base_path, "artifacts/data")
            CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
                "root_dir"
            ] = split_data_root_dir
            split_data_filepath = os.path.join(
                split_data_root_dir, split_data_filename + ".npz"
            )
            CONFIG[PROJECT_NAME]["artifacts"]["split-data"][
                "filepath"
            ] = split_data_filepath
            print("↑↑↑ Pulling MNIST dataset from keras...")
            (X_tr, y_tr), (X_te, y_te) = mnist.load_data()
            # We rather want to split into train / val / test
            X = np.concatenate([X_tr, X_te])
            y = np.concatenate([y_tr, y_te])

            datasets = split_data(X, y, args.train_val_test_size)
            print("Normalizing and reshaping...")
            datasets.X_train = (datasets.X_train / 255) - 0.5
            datasets.X_train = np.expand_dims(datasets.X_train, axis=3)
            datasets.X_val = (datasets.X_val / 255) - 0.5
            datasets.X_val = np.expand_dims(datasets.X_val, axis=3)
            datasets.X_test = (datasets.X_test / 255) - 0.5
            datasets.X_test = np.expand_dims(datasets.X_test, axis=3)
            print("↓↓↓ Saving split data as artifact...")
            np.savez_compressed(
                os.path.join(split_data_filepath),
                X_train=datasets.X_train,
                y_train=datasets.y_train,
                X_val=datasets.X_val,
                y_val=datasets.y_val,
                X_test=datasets.X_test,
                y_test=datasets.y_test,
            )
            split_data_artifact = wandb.Artifact(
                split_data_filename,
                type="dataset",
                description="artifact consisting of train / val / test datasets. It comes with a table on targets to check class stratification in wandb",  # noqa: E501
                metadata={
                    "X_train shape": datasets.X_train.shape,
                    "y_train shape": datasets.y_train.shape,
                    "X_val shape": datasets.X_val.shape,
                    "y_val shape": datasets.y_val.shape,
                    "X_test shape": datasets.X_test.shape,
                    "y_test shape": datasets.y_test.shape,
                },
            )
            split_data_artifact.add_file(local_path=split_data_filepath)
            # create wandb table on targets
            labels_df = build_labels_df(datasets)
            table = wandb.Table(dataframe=labels_df)
            split_data_artifact.add(table, "split_labels")
            run.log_artifact(split_data_artifact)

    else:
        print("Retrieve datasets from wandb")
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
            run.use_artifact(split_data_filename + ":latest").download(
                split_data_root_dir
            )
            datasets = np.load(split_data_filepath, allow_pickle=True)

    if not is_dataclass(datasets):
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
    print("test examples: ", len(datasets.X_test))

    # save updates to CONFIG file
    with open(config_filepath, "w") as f:
        json.dump(CONFIG, f, indent=2)

    return datasets


def split_data(X, y, train_val_test_size):
    split_list = list(map(float, train_val_test_size.split(" ")))
    assert len(split_list) == 3
    CONFIG[PROJECT_NAME]["artifacts"]["split-data"]["train_val_test_size"] = split_list
    split_list = np.divide(split_list, 100.0)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        shuffle=True,
        test_size=split_list[2],
        stratify=y,
    )
    assert np.all(np.unique(y_trainval) == np.unique(y_test))
    val_size = split_list[1] / (split_list[0] + split_list[1])

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_size,
        stratify=y_trainval,
    )
    assert np.all(np.unique(y_train) == np.unique(y_val))

    datasets = DataSets(X_train, y_train, X_val, y_val, X_test, y_test)

    return datasets


def build_labels_df(datasets):
    y_train_df = pd.DataFrame(datasets.y_train, columns=["label"])
    y_train_df["stage"] = "train"
    y_val_df = pd.DataFrame(datasets.y_val, columns=["label"])
    y_val_df["stage"] = "valid"
    y_test_df = pd.DataFrame(datasets.y_test, columns=["label"])
    y_test_df["stage"] = "test"
    labels_df = pd.concat([y_train_df, y_val_df, y_test_df])

    return labels_df


@dataclass
class DataSets:
    X_train: float
    y_train: float
    X_val: float
    y_val: float
    X_test: float
    y_test: float


if __name__ == "__main__":
    # python app/pipelines/data_prep.py --build_datasets
    # Parse args
    docstring = """When running this script as main you can specify the filepath for the raw data to be prepared """  # noqa: E501
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--build_datasets", default=False, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--train_val_test_size", nargs="+", default="70 20 10")
    args = parser.parse_args()

    wandb_key = os.environ.get("WANDB_API_KEY")
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("=== Data preparation pipeline ===")
        data_preparation_pipeline(args)
        print("=== Finished ===")
