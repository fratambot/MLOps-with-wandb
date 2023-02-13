import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import wandb

from dotenv import load_dotenv
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras

# load env variables
load_dotenv()

# we use /app folder as base_path
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# and we add it at the beginning of PYTHONPATH otherwise we
# cannot import from modules at the same level
sys.path.insert(0, base_path)

# local import
from utils.data import datasets_loader  # noqa: E402


def evaluate():
    config_filepath = os.path.join(base_path, "config.json")
    global LOCAL_CONFIG
    LOCAL_CONFIG = json.load(open(config_filepath))
    wandb.login()
    global PROJECT_NAME
    PROJECT_NAME = os.environ.get("WANDB_PROJECT")
    datasets = datasets_loader(LOCAL_CONFIG, PROJECT_NAME)

    with wandb.init(
        project=PROJECT_NAME, job_type="evaluation", tags=["evaluate"]
    ) as run:
        # print("my default config in wandb = ", config)
        model_root_dir = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"]["root_dir"]
        model_filepath = LOCAL_CONFIG[PROJECT_NAME]["artifacts"]["models"]["filepath"]
        artifact_path = os.environ.get("WANDB_MODEL_EVAL")
        artifact = run.use_artifact(artifact_path, type="model")
        producer_run = artifact.logged_by()
        wandb.config.update(producer_run.config)
        config = wandb.config
        artifact.download(model_root_dir)
        model = keras.models.load_model(model_filepath)
        print("artifact model successfully loaded")
        print("Evaluating the model on the validation and test set...")
        val_metrics = model.evaluate(
            datasets.X_val,
            datasets.y_val,
            batch_size=config.batch_size,
            return_dict=True,
        )
        test_metrics = model.evaluate(
            datasets.X_test,
            datasets.y_test,
            batch_size=config.batch_size,
            return_dict=True,
        )
        run.log(
            {
                "val_loss": val_metrics["loss"],
                "val_sparse_categorical_crossentropy": val_metrics[
                    "sparse_categorical_crossentropy"
                ],
                "test_loss": test_metrics["loss"],
                "test_sparse_categorical_crossentropy": test_metrics[
                    "sparse_categorical_crossentropy"
                ],
            }
        )
        print("Making predictions on the validation and test set...")
        val_preds = model.predict(datasets.X_val, batch_size=config.batch_size)
        test_preds = model.predict(datasets.X_test, batch_size=config.batch_size)
        test_images_path = generate_images(datasets)
        # build confusion_matrices
        val_cm_path, test_cm_path = make_confusion_matrices(
            val_preds, test_preds, datasets
        )
        # val_hist_path, test_hist_path = make_histograms(
        #   val_preds, test_preds, datasets
        # )
        run.log(
            {
                "confusion_matrices": [
                    wandb.Image(f"{val_cm_path}.png", caption="val_cm"),
                    wandb.Image(f"{test_cm_path}.png", caption="test_cm"),
                ]
            }
        )
        # val_df = pd.DataFrame([
        #     datasets.y_val,
        #     np.argmax(val_preds, axis=1),
        # ]).T
        # val_df.columns=["val_true", "val_pred"]
        test_df = pd.DataFrame(
            [
                datasets.y_test[:10],
                np.argmax(test_preds, axis=1)[:10],
            ]
        ).T
        test_df.columns = ["test_true", "test_pred"]
        # val_table = wandb.Table(dataframe=val_df)
        # run.log({"validation_table": val_table})
        test_table = wandb.Table(columns=["image", "test_true", "test_pred"])
        for i in range(10):
            img_path = os.path.join(test_images_path, f"test_image_{i}.png")
            test_table.add_data(
                wandb.Image(img_path),
                test_df.test_true[i],
                test_df.test_pred[i],
            )
        run.log({"test_table": test_table})

    return


def make_confusion_matrices(val_preds, test_preds, datasets):
    val_disp = ConfusionMatrixDisplay.from_predictions(
        y_true=datasets.y_val, y_pred=np.argmax(val_preds, axis=1)
    )
    fig = val_disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    val_disp.ax_.set_title("Confusion matrix on validation set")
    val_cm_path = os.path.join(base_path, "artifacts/figures", "val_cm")
    fig.savefig(val_cm_path, bbox_inches="tight", facecolor="w")

    test_disp = ConfusionMatrixDisplay.from_predictions(
        y_true=datasets.y_test, y_pred=np.argmax(test_preds, axis=1)
    )
    fig = test_disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    test_disp.ax_.set_title("Confusion matrix on test set")
    test_cm_path = os.path.join(base_path, "artifacts/figures", "test_cm")
    fig.savefig(test_cm_path, bbox_inches="tight", facecolor="w")

    return val_cm_path, test_cm_path


def make_histograms(val_preds, test_preds, datasets):
    print(datasets.X_val.value_counts)
    return


def generate_images(datasets):
    for i, test_example in enumerate(datasets.X_test[:10]):
        fig, axs = plt.subplots()
        axs.imshow(test_example, cmap=plt.get_cmap("gray"))
        img_name = f"test_image_{i}.png"
        img_path = os.path.join(base_path, "artifacts/data/images/test", img_name)
        fig.savefig(img_path, bbox_inches="tight", facecolor="w")

    test_images_path = os.path.join(base_path, "artifacts/data/images/test")

    return test_images_path


if __name__ == "__main__":
    # python app/pipelines/evaluate.py

    wandb_key = os.environ.get("WANDB_API_KEY")
    # TODO: add model_eval env
    if wandb_key is None:
        print(
            "ERROR: Weights and Biases integration failed. You need a wandb account to run this script"  # noqa: E501
        )
    else:
        print("=== Model evaluation pipeline ===")
        evaluate()
        print("=== Finished ===")
