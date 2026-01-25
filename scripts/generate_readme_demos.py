import os
import argparse
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate README demo visuals.")
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
        "--out-file", default="docs/demo_preview.png", help="Output image path"
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

    label = "Bad (Blur)" if pred_idx == 0 else "Good (Sharp)"

    return img_arr.astype(np.uint8), label, conf


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)
    out_file = Path(args.out_file)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    input_shape = model.input_shape
    target_size = (384, 384)
    if input_shape and input_shape[1]:
        target_size = (input_shape[1], input_shape[2])

    good_dir = data_dir / "good"
    bad_dir = data_dir / "bad"

    good_files = list(good_dir.glob("*"))
    bad_files = list(bad_dir.glob("*"))


    print("Selecting demo images...")

    demo_bad = []
    random.shuffle(bad_files)
    for f in bad_files:
        if len(demo_bad) >= 3:
            break
        img, label, conf = load_and_predict(model, f, target_size)
        if label == "Bad (Blur)" and conf > 0.8:
            demo_bad.append((img, f"{label}\nConf: {conf:.0%}"))

    demo_good = []
    random.shuffle(good_files)
    for f in good_files:
        if len(demo_good) >= 3:
            break
        img, label, conf = load_and_predict(model, f, target_size)
        if label == "Good (Sharp)" and conf > 0.8:
            demo_good.append((img, f"{label}\nConf: {conf:.0%}"))

    images = demo_bad + demo_good

    rows, cols = 2, 3
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()

    for i, (img, txt) in enumerate(images):
        axes[i].imshow(img)
        axes[i].set_title(
            txt, color="green" if "Good" in txt else "red", fontweight="bold"
        )
        axes[i].axis("off")

    fig.suptitle("Model Inference Demo", fontsize=16)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_file)
    print(f"Saved demo image to {out_file}")


if __name__ == "__main__":
    main()
