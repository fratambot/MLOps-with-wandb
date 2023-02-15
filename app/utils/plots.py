import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm


def make_labels_histogram(base_path, datasets, train_val_test_size):
    split_list = list(map(float, train_val_test_size.split(" ")))
    assert len(split_list) == 3
    rescaled_frames = []
    for i, dataset_name in enumerate(["train", "val", "test"]):
        subset = getattr(datasets, "y_" + dataset_name)
        value_counts = get_value_counts(subset, dataset_name)
        rescaled = value_counts[dataset_name] * (100.0 / split_list[i])
        rescaled_frames.append(rescaled)

    rescaled_df = pd.concat(rescaled_frames, axis=1)
    rescaled_df.plot(
        kind="bar",
        stacked=False,
        title="Rescaled distribution of labels for the 3 datasets",
        ylabel="Rescaled arbitrary counts",
        ylim=(0, 1000),
        rot=0,
    )
    histogram_path = os.path.join(base_path, "artifacts/figures", "labels_histogram")
    plt.savefig(histogram_path, bbox_inches="tight", facecolor="w")

    return histogram_path


def get_value_counts(dataset, name):
    unique, counts = np.unique(dataset, return_counts=True)
    zipped = list(zip(unique, counts))
    value_counts = pd.DataFrame(zipped, columns=["label", f"{name}"])
    value_counts.set_index("label", inplace=True)
    return value_counts


def make_confusion_matrix(base_path, y_true, y_pred, dataset_label):
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred)
    fig = disp.ax_.get_figure()
    fig.set_figwidth(10)
    fig.set_figheight(10)
    disp.ax_.set_title(f"Confusion matrix on {dataset_label} set")
    cm_path = os.path.join(base_path, "artifacts/figures", f"{dataset_label}_cm")
    fig.savefig(cm_path, bbox_inches="tight", facecolor="w")
    plt.close(fig)

    return cm_path


def generate_images(base_path, data, dataset_label):
    print(f"Generating {dataset_label} images from arrays. This might take a while...")
    images_folder_path = os.path.join(base_path, "artifacts/data/images", dataset_label)
    for i, example in enumerate(tqdm(data)):
        fig, axs = plt.subplots()
        axs.imshow(np.squeeze(example), cmap=plt.get_cmap("gray"))
        img_name = f"{dataset_label}_image_{i}.png"
        img_path = os.path.join(images_folder_path, img_name)
        fig.savefig(img_path, bbox_inches="tight", facecolor="w")
        plt.close(fig)

    return images_folder_path
