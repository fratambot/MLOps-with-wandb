import numpy as np
import wandb

from dataclasses import dataclass


def datasets_loader(LOCAL_CONFIG, PROJECT_NAME):
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
        print("test examples: ", len(datasets.X_test))

    return datasets


@dataclass
class DataSets:
    X_train: float
    y_train: float
    X_val: float
    y_val: float
    X_test: float
    y_test: float
