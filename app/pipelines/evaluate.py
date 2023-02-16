import argparse
import numpy as np
import os
import pandas as pd
import sys
import wandb

from dotenv import load_dotenv

from types import SimpleNamespace
from tensorflow import keras

# load env variables
load_dotenv()

# use "*/app" folder as base_path for local imports
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, base_path)

# local import
from utils.data import datasets_loader  # noqa: E402
from utils.plots import make_confusion_matrix, generate_images  # noqa: E402

default_config = SimpleNamespace(generate_images=True)


def parse_args():
    docstring = """Overriding default argments"""
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--generate_images",
        default=default_config.generate_images,
        action=argparse.BooleanOptionalAction,
        help="generate validation and test examples images",
    )
    args = parser.parse_args()
    return args


def evaluate(config):
    wandb.login()
    artifact_folder_path = os.path.join(base_path, "artifacts", "data")
    artifact_name = "split-data"
    artifact_extension = ".npz"
    artifact_filepath = os.path.join(
        artifact_folder_path, artifact_name + artifact_extension
    )
    datasets = datasets_loader(
        config.project, artifact_folder_path, artifact_name, artifact_filepath
    )

    with wandb.init(
        project=config.project, job_type=config.job_type, tags=["evaluate"]
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
        print("Evaluating the model on the validation and test set...")
        val_metrics = model.evaluate(
            datasets.X_val,
            datasets.y_val,
            batch_size=final_config.batch_size,
            return_dict=True,
        )
        test_metrics = model.evaluate(
            datasets.X_test,
            datasets.y_test,
            batch_size=final_config.batch_size,
            return_dict=True,
        )
        run.log(
            {
                "val_loss": val_metrics["loss"],
                "val_sparse_categorical_accuracy": val_metrics[
                    "sparse_categorical_accuracy"
                ],
                "test_loss": test_metrics["loss"],
                "test_sparse_categorical_accuracy": test_metrics[
                    "sparse_categorical_accuracy"
                ],
            }
        )
        print("Making predictions on the validation and test set...")
        val_preds = model.predict(datasets.X_val, batch_size=final_config.batch_size)
        test_preds = model.predict(datasets.X_test, batch_size=final_config.batch_size)
        # build confusion_matrices
        val_cm_path = make_confusion_matrix(
            base_path, datasets.y_val, np.argmax(val_preds, axis=1), "validation"
        )
        test_cm_path = make_confusion_matrix(
            base_path, datasets.y_test, np.argmax(test_preds, axis=1), "test"
        )
        # log to wand as media
        run.log(
            {
                "confusion_matrices": [
                    wandb.Image(f"{val_cm_path}.png", caption="val_cm"),
                    wandb.Image(f"{test_cm_path}.png", caption="test_cm"),
                ]
            }
        )
        # Build tables for wandb
        test_df = build_pred_df(datasets.y_test, np.argmax(test_preds, axis=1), "test")
        val_df = build_pred_df(datasets.y_val, np.argmax(val_preds, axis=1), "val")
        if final_config.generate_images:
            val_table = wandb.Table(columns=["image", "val_true", "val_pred"])
            val_images_folder_path = generate_images(
                base_path, datasets.X_val, "validation"
            )
            for i in range(len(datasets.X_val)):
                img_path = os.path.join(
                    val_images_folder_path, f"validation_image_{i}.png"
                )
                val_table.add_data(
                    wandb.Image(img_path),
                    val_df.val_true[i],
                    val_df.val_pred[i],
                )

            test_table = wandb.Table(columns=["image", "test_true", "test_pred"])
            test_images_folder_path = generate_images(
                base_path, datasets.X_test, "test"
            )
            for i in range(len(datasets.X_test)):
                img_path = os.path.join(test_images_folder_path, f"test_image_{i}.png")
                test_table.add_data(
                    wandb.Image(img_path),
                    test_df.test_true[i],
                    test_df.test_pred[i],
                )
        else:
            test_table = wandb.Table(
                dataframe=test_df, columns=["test_true", "test_pred"]
            )
            val_table = wandb.Table(dataframe=val_df, columns=["val_true", "val_pred"])

        run.log({"val_table": val_table})
        run.log({"test_table": test_table})

    return


def build_pred_df(y_true, y_pred, label):
    df = pd.DataFrame([y_true, y_pred]).T
    df.columns = [f"{label}_true", f"{label}_pred"]

    return df


if __name__ == "__main__":
    # python app/pipelines/evaluate.py
    # python app/pipelines/evaluate.py --no-generate_images

    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_entity = os.environ.get("WANDB_ENTITY")
    model_to_evaluate = os.environ.get("WANDB_MODEL_EVAL")
    if (
        wandb_key is None
        or wandb_project is None
        or wandb_entity is None
        or model_to_evaluate is None
    ):
        print(
            "ERROR: You need to set the WANDB_API_KEY, WANDB_PROJECT, "
            "WANDB_ENTITY as well as WANDB_MODEL_EVAL env variables to use this script"
        )
    else:
        print("=== Model evaluation pipeline ===")
        args = parse_args()
        config = SimpleNamespace(
            project=wandb_project,
            entity=wandb_entity,
            model_id=model_to_evaluate,
            job_type="evaluation",
        )
        # update default_config with args
        vars(config).update(vars(args))
        evaluate(config)
        print("=== Finished ===")
