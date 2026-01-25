import os
import cv2
import numpy as np

from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.adapters import classify

MODEL_PATH = "quality_model_large_edgetpu_edgetpu.tflite"
TEST_IMAGE_DIR = "prepared/data_formatted_native_320/val"
INPUT_SIZE = 384


def get_5_crops(img, size=INPUT_SIZE):
    h, w, c = img.shape

    if h < size or w < size:
        scale = size / min(h, w)
        img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
        h, w, c = img.shape

    center_h, center_w = h // 2, w // 2
    half = size // 2

    crops = []
    crops.append(
        img[center_h - half : center_h + half, center_w - half : center_w + half]
    )
    crops.append(img[0:size, 0:size])
    crops.append(img[0:size, w - size : w])
    crops.append(img[h - size : h, 0:size])
    crops.append(img[h - size : h, w - size : w])

    return crops


def predict_smart(interpreter, image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    crops = get_5_crops(img, INPUT_SIZE)
    scores = []

    for patch in crops:
        common.set_input(interpreter, patch)

        interpreter.invoke()

        classes = classify.get_classes(interpreter, top_k=2)

        bad_score = 0.0
        for c in classes:
            if c.id == 0:
                bad_score = c.score
                break

        scores.append(bad_score)

    avg_score = np.mean(scores)
    return avg_score


def main():
    print(f"Loading TPU Model: {MODEL_PATH}...")

    try:
        interpreter = edgetpu.make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        print("TPU Loaded Successfully!")
    except Exception as e:
        print(f"Error loading TPU: {e}")
        print(
            "Tip: Make sure the USB Accelerator is plugged in and 'edgetpu_runtime' is installed."
        )
        return

    print("-" * 50)
    print(f"{'Filename':<25} | {'Bad Confidence':<15} | {'Verdict'}")
    print("-" * 50)

    THRESHOLD = 0.50

    bad_dir = os.path.join(TEST_IMAGE_DIR, "bad")
    print(">>> Testing BAD Images")
    if os.path.exists(bad_dir):
        for f in os.listdir(bad_dir)[:10]:
            path = os.path.join(bad_dir, f)
            score = predict_smart(interpreter, path)
            status = "CORRECT" if score > THRESHOLD else "WRONG"
            print(f"{f[:23]:<25} | {score*100:.1f}%          | {status}")

    print("-" * 50)

    good_dir = os.path.join(TEST_IMAGE_DIR, "good")
    print(">>> Testing GOOD Images")
    if os.path.exists(good_dir):
        for f in os.listdir(good_dir)[:10]:
            path = os.path.join(good_dir, f)
            score = predict_smart(interpreter, path)
            status = "CORRECT" if score < THRESHOLD else "WRONG"
            print(f"{f[:23]:<25} | {score*100:.1f}%          | {status}")


if __name__ == "__main__":
    main()
