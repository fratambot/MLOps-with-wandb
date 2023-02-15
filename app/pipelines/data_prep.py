import argparse
import numpy as np
import pandas as pd
import os
import sys
import wandb

from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from tensorflow import keras

# load env variables
load_dotenv()

# use "*/app" folder as base_path for local imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# local imports
from utils.data import DataSets  # noqa: E402
from utils.plots import make_labels_histogram  # noqa: E402


def parse_args():
    docstring = """This pipeline will build a train / val / test dataset from the keras MNIST dataset """  # noqa: E501
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fraction", default=10, type=int)
    parser.add_argument("--train_val_test_size", nargs="+", default="70 20 10")
    args = parser.parse_args()

    return args


def data_preparation_pipeline(args):
    wandb.login()
    # global PROJECT_NAME
    # PROJECT_NAME = os.environ.get("WANDB_PROJECT")
    with wandb.init(project=args.project, job_type="data-split") as run:
        artifact_folder_name = "data"
        artifact_name = "split-data"
        artifact_extension = ".npz"
        artifact_filepath = os.path.join(
            base_path,
            "artifacts",
            artifact_folder_name,
            artifact_name + artifact_extension,
        )
        print("↑↑↑ Pulling MNIST dataset from keras...")
        (X_tr, y_tr), (X_te, y_te) = keras.datasets.mnist.load_data()
        X = np.concatenate([X_tr, X_te])
        y = np.concatenate([y_tr, y_te])
        # Use only a fraction of the dataset
        (X, y) = sample(X, y, args.fraction)
        print("X sample = ", X.shape)
        print("y sample = ", y.shape)
        # We rather want to split into train / val / test
        datasets = split_data(X, y, args.train_val_test_size)
        print("Normalizing and reshaping...")
        datasets = normalize_and_reshape(datasets)
        print("↓↓↓ Saving split data as artifact...")
        np.savez_compressed(
            os.path.join(artifact_filepath),
            X_train=datasets.X_train,
            y_train=datasets.y_train,
            X_val=datasets.X_val,
            y_val=datasets.y_val,
            X_test=datasets.X_test,
            y_test=datasets.y_test,
        )
        split_data_artifact = wandb.Artifact(
            artifact_name,
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
        split_data_artifact.add_file(local_path=artifact_filepath)
        # make histogram of labels distributions for the 3 sets
        hist_path = make_labels_histogram(base_path, datasets, args.train_val_test_size)
        # log to wand as media
        run.log(
            {
                "labels_distribution": wandb.Image(
                    f"{hist_path}.png", caption="rescaled_labels_distribution"
                )
            }
        )
        # create wandb table on labels
        labels_df = build_labels_df(datasets)
        split_table = wandb.Table(dataframe=labels_df)
        # log table to wandb
        run.log({"split_table": split_table})
        # split_data_artifact.add(split_table, "split_labels")
        run.log_artifact(split_data_artifact)

    print("training examples: ", len(datasets.X_train))
    print("validation examples: ", len(datasets.X_val))
    print("test examples: ", len(datasets.X_test))

    return datasets


def sample(X, y, fraction):
    print(f"Sampling {fraction} % of the full dataset")
    X_sample, _, y_sample, _ = train_test_split(
        X, y, shuffle=True, train_size=fraction / 100.0, stratify=y
    )

    return X_sample, y_sample


def split_data(X, y, train_val_test_size):
    print(f"Splitting data into train / val / test : {train_val_test_size} %")
    split_list = list(map(float, train_val_test_size.split(" ")))
    assert len(split_list) == 3

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


def normalize_and_reshape(datasets):
    for dataset in ["X_train", "X_val", "X_test"]:
        subset = getattr(datasets, dataset)
        subset = (subset / 255) - 0.5
        subset = np.expand_dims(subset, axis=3)
        setattr(datasets, dataset, subset)

    return datasets


def build_labels_df(datasets):
    frames = []
    for dataset in ["y_train", "y_val", "y_test"]:
        df = pd.DataFrame(getattr(datasets, dataset), columns=["label"])
        df["stage"] = dataset
        frames.append(df)

    labels_df = pd.concat(frames)

    return labels_df


if __name__ == "__main__":
    # python app/pipelines/data_prep.py

    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    if wandb_key is None or wandb_project is None:
        print(
            "ERROR: You need to set the WANDB_API_KEY and"
            "WANDB_PROJECT env variables to use this script"
        )
    else:
        print("=== Data preparation pipeline ===")
        args = parse_args()
        args.project = wandb_project
        data_preparation_pipeline(args)
        print("=== Finished ===")
