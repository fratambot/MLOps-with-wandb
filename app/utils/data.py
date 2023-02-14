import numpy as np
import wandb

from dataclasses import dataclass


def datasets_loader(project_name, art_folder_path, art_name, art_filepath):
    print("↑↑↑ Loading latest datasets from wandb...")
    with wandb.init(project=project_name, job_type="load-data") as run:
        run.use_artifact(art_name + ":latest").download(art_folder_path)
        datasets = np.load(art_filepath, allow_pickle=True)
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
