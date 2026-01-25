import os
import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate failure illustration grid.")
    parser.add_argument(
        "--model-path",
        default="models/data_formatted_native_384/quality_model_max_best.h5",
        help="Path to .h5 model",
    )
    parser.add_argument(
        "--data-dir",
        default="prepared/data_formatted_native_384/val",
        help="Validation folder path",
    )
    parser.add_argument(
        "--out-file",
        default="devlog/model_visuals_final/failure_analysis_grid.png",
        help="Output image path",
    )
    return parser.parse_args()


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image


def load_and_predict(model, image_path, target_size=(384, 384)):
    img_raw = tf.keras.utils.load_img(image_path, target_size=target_size)
    img_arr = tf.keras.utils.img_to_array(img_raw)
    img_pre = preprocess(img_arr)
    img_batch = tf.expand_dims(img_pre, axis=0)

    probs = model.predict(img_batch, verbose=0)
    pred_idx = np.argmax(probs)
    conf = np.max(probs)

    return img_arr.astype(np.uint8), pred_idx, conf


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    out_file = Path(args.out_file)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    input_shape = model.input_shape
    target_size = (384, 384)
    if input_shape and input_shape[1]:
        target_size = (input_shape[1], input_shape[2])
    print(f"Using target size: {target_size}")

    good_dir = data_dir / "good"
    bad_dir = data_dir / "bad"

    good_files = (
        list(good_dir.glob("*.jpg"))
        + list(good_dir.glob("*.png"))
        + list(good_dir.glob("*.bmp"))
    )
    bad_files = (
        list(bad_dir.glob("*.jpg"))
        + list(bad_dir.glob("*.png"))
        + list(bad_dir.glob("*.bmp"))
    )

    print(f"Found {len(good_files)} 'good' images and {len(bad_files)} 'bad' images.")

    fp_samples = []
    tn_samples = []
    tp_samples = []

    random.shuffle(good_files)
    random.shuffle(bad_files)

    print("Scanning for samples...")

    for f in good_files[:200]:
        if len(fp_samples) >= 4:
            break
        img, pred, conf = load_and_predict(model, f, target_size)
        if pred == 0:

            fp_samples.append((img, pred, conf, "GT: Good -> Pred: Bad"))
        elif len(tp_samples) < 4:
            tp_samples.append((img, pred, conf, "GT: Good -> Pred: Good"))

    for f in bad_files[:100]:
        if len(tn_samples) >= 4:
            break
        img, pred, conf = load_and_predict(model, f, target_size)
        if pred == 0:
            tn_samples.append((img, pred, conf, "GT: Bad -> Pred: Bad"))

    final_samples = fp_samples[:4] + tp_samples[:4] + tn_samples[:4]

    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()

    row_titles = [
        "Incorrectly Tagged as Bad (False Positives)",
        "Correctly Tagged as Good (True Positives)",
        "Correctly Tagged as Bad (True Negatives)",
    ]

    for i, ax in enumerate(axes):
        if i < len(final_samples):
            img, pred, conf, label = final_samples[i]
            ax.imshow(img)
            ax.axis("off")

            color = "red" if "Good -> Pred: Bad" in label else "green"
            if "Bad -> Pred: Bad" in label:
                color = "blue"

            ax.set_title(
                f"{label}\nConf: {conf:.2f}",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.axis("off")

    fig.suptitle(
        "Model Failure Analysis: Subjectivity & Misclassification", fontsize=20
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")


if __name__ == "__main__":
    main()
