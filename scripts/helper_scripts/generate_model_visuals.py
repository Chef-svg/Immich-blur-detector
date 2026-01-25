import argparse
import os
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate model evaluation visuals for devlog uploads."
    )
    parser.add_argument(
        "--model-path",
        default="models/data_formatted_native_384/quality_model_max_best.h5",
        help="Path to the trained Keras model (.h5).",
    )
    parser.add_argument(
        "--data-dir",
        default="prepared/data_formatted_native_384",
        help="Dataset root that contains val/ folder.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override square image size. If omitted, inferred from model input.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for inference."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="How many sample images to include in grids.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for generated images.",
    )
    return parser.parse_args()


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image, label


def get_image_size_from_model(model):
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    if input_shape and len(input_shape) >= 3:
        height, width = input_shape[1], input_shape[2]
        if height and width:
            return int(height), int(width)
    return 384, 384


def ensure_out_dir(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_confusion_matrix(cm, class_names, out_path, normalize=True):
    if normalize:
        cm = cm.astype("float") / np.maximum(cm.sum(axis=1)[:, np.newaxis], 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix" + (" (Normalized)" if normalize else ""),
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_image_for_plot(path, target_size):
    image = tf.keras.utils.load_img(path, target_size=target_size)
    return tf.keras.utils.img_to_array(image).astype(np.uint8)


def plot_sample_grid(
    title,
    file_paths,
    y_true,
    y_pred,
    y_pred_probs,
    class_names,
    out_path,
    samples=24,
    seed=42,
):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(file_paths), size=min(samples, len(file_paths)), replace=False
    )

    cols = 6
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(indices):
            ax.axis("off")
            continue
        idx = indices[ax_idx]
        img = load_image_for_plot(file_paths[idx], target_size=(256, 256))
        pred_label = class_names[y_pred[idx]]
        true_label = class_names[y_true[idx]]
        conf = float(np.max(y_pred_probs[idx]))
        ax.imshow(img.astype(np.uint8))
        ax.axis("off")
        ax.set_title(
            f"P: {pred_label} ({conf:.2f})\nT: {true_label}",
            fontsize=9,
            color=("green" if pred_label == true_label else "red"),
        )

    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_grid(
    file_paths,
    y_true,
    y_pred,
    y_pred_probs,
    class_names,
    out_path,
    samples=24,
):
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        return

    selected = mis_idx[: min(samples, len(mis_idx))]
    cols = 6
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(selected):
            ax.axis("off")
            continue
        idx = selected[ax_idx]
        img = load_image_for_plot(file_paths[idx], target_size=(256, 256))
        pred_label = class_names[y_pred[idx]]
        true_label = class_names[y_true[idx]]
        conf = float(np.max(y_pred_probs[idx]))
        ax.imshow(img.astype(np.uint8))
        ax.axis("off")
        ax.set_title(
            f"P: {pred_label} ({conf:.2f})\nT: {true_label}", fontsize=9, color="red"
        )

    fig.suptitle("Misclassified Samples", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_histogram(y_true, y_pred, y_pred_probs, out_path):
    max_probs = np.max(y_pred_probs, axis=1)
    correct_mask = y_true == y_pred

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        max_probs[correct_mask],
        bins=20,
        alpha=0.7,
        label="Correct",
        color="green",
    )
    ax.hist(
        max_probs[~correct_mask],
        bins=20,
        alpha=0.7,
        label="Incorrect",
        color="red",
    )
    ax.set_title("Prediction Confidence Histogram")
    ax.set_xlabel("Max Softmax Probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true, y_pred_probs, class_names, out_path):
    n_classes = len(class_names)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_onehot[:, i], y_pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ax.plot(
            fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})"
        )

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(y_true, y_pred_probs, class_names, out_path):
    n_classes = len(class_names)
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=n_classes)

    fig, ax = plt.subplots(figsize=(8, 6))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_probs[:, i]
        )
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"Class {class_names[i]} (AUC = {pr_auc:.2f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)

    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)

    if args.image_size is None:
        image_size = get_image_size_from_model(model)
    else:
        image_size = (args.image_size, args.image_size)

    out_dir = args.out_dir
    if out_dir is None:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = data_dir.parent / "devlog" / f"model_visuals_{stamp}"
    out_dir = ensure_out_dir(out_dir)

    print(f"Preparing validation dataset from: {data_dir}")
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        image_size=image_size,
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = val_ds_raw.class_names
    file_paths = val_ds_raw.file_paths

    val_ds = val_ds_raw.map(preprocess).prefetch(tf.data.AUTOTUNE)

    print("Running inference...")
    y_pred_probs = model.predict(val_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for _, labels in val_ds_raw:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)

    print("Generating report...")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    report_path = Path(out_dir) / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(
        cm,
        class_names,
        out_path=Path(out_dir) / "confusion_matrix.png",
        normalize=False,
    )
    plot_confusion_matrix(
        cm,
        class_names,
        out_path=Path(out_dir) / "confusion_matrix_normalized.png",
        normalize=True,
    )

    plot_sample_grid(
        "Random Validation Samples",
        file_paths,
        y_true,
        y_pred,
        y_pred_probs,
        class_names,
        out_path=Path(out_dir) / "sample_predictions_grid.png",
        samples=args.samples,
    )

    plot_misclassified_grid(
        file_paths,
        y_true,
        y_pred,
        y_pred_probs,
        class_names,
        out_path=Path(out_dir) / "misclassified_grid.png",
        samples=args.samples,
    )

    plot_confidence_histogram(
        y_true,
        y_pred,
        y_pred_probs,
        out_path=Path(out_dir) / "prediction_confidence_hist.png",
    )

    plot_roc_curve(
        y_true,
        y_pred_probs,
        class_names,
        out_path=Path(out_dir) / "roc_curve.png",
    )

    plot_pr_curve(
        y_true,
        y_pred_probs,
        class_names,
        out_path=Path(out_dir) / "pr_curve.png",
    )

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
