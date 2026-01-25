import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    auc,
)


BASE_DIR = Path(__file__).parent.parent.parent
DATA_ROOT = BASE_DIR / "prepared/data_formatted_native_384"
MODEL_DIR = BASE_DIR / "models/data_formatted_native_384"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a TFLite model on prepared quality dataset."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to a TFLite model. If omitted, auto-detects best candidate.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for class 0 (bad) score.",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="Find the best threshold on the evaluated split.",
    )
    parser.add_argument(
        "--threshold-metric",
        default="f1",
        choices=["f1", "youden"],
        help="Metric for auto-threshold selection.",
    )
    return parser.parse_args()


def resolve_model_path(model_path_arg: Optional[str]) -> Tuple[Path, bool]:
    if model_path_arg:
        path = Path(model_path_arg)
        return path, "_edgetpu" in path.stem

    candidates = [
        MODEL_DIR / "quality_model_max_int8_edgetpu.tflite",
        MODEL_DIR / "quality_model_max_edgetpu_edgetpu.tflite",
        MODEL_DIR / "quality_model_max_int8.tflite",
    ]
    for cand in candidates:
        if cand.exists():
            return cand, "_edgetpu" in cand.stem

    raise FileNotFoundError("No suitable TFLite model found.")


def load_interpreter(model_path: Path, use_edgetpu: bool):
    try:
        if use_edgetpu:
            from pycoral.utils import edgetpu

            interpreter = edgetpu.make_interpreter(str(model_path))
        else:
            import tensorflow as tf

            interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def _resize_and_normalize(img: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return img


def _quantize_input(img: np.ndarray, input_details) -> np.ndarray:
    dtype = input_details[0]["dtype"]
    scale, zero_point = input_details[0]["quantization"]

    if dtype in (np.uint8, np.int8):
        if scale == 0:
            return img.astype(dtype)
        img = img / scale + zero_point
        info = np.iinfo(dtype)
        img = np.clip(img, info.min, info.max).astype(dtype)
        return img

    return img.astype(dtype)


def _dequantize_output(output: np.ndarray, output_details) -> np.ndarray:
    dtype = output_details[0]["dtype"]
    scale, zero_point = output_details[0]["quantization"]
    if dtype in (np.uint8, np.int8):
        return (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def predict(interpreter, img_path: str) -> Optional[float]:
    img = cv2.imread(img_path)
    if img is None:
        return None

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    height, width = int(input_shape[1]), int(input_shape[2])

    img = _resize_and_normalize(img, (width, height))
    img = _quantize_input(img, input_details)

    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
    interpreter.invoke()

    output_index = output_details[0]["index"]
    output = interpreter.get_tensor(output_index)[0]
    output = _dequantize_output(output, output_details)

    if output.shape[0] < 2:
        return float(output[0])

    return float(output[0])


def main():
    args = parse_args()

    model_path, use_edgetpu = resolve_model_path(args.model_path)
    interpreter = load_interpreter(model_path, use_edgetpu)

    eval_dir = DATA_ROOT / args.split
    if not eval_dir.exists():
        raise FileNotFoundError(f"Split not found: {eval_dir}")

    y_true = []
    scores = []

    mode = "EdgeTPU" if use_edgetpu else "CPU"
    print(f"Evaluating {str(model_path)} on {str(eval_dir)} using {mode}...")
    start_time = time.time()

    bad_path = os.path.join(str(eval_dir), "bad")
    for f in os.listdir(bad_path):
        if not f.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        score = predict(interpreter, os.path.join(bad_path, f))
        if score is not None:
            y_true.append(0)
            scores.append(score)

    good_path = os.path.join(str(eval_dir), "good")
    for f in os.listdir(good_path):
        if not f.lower().endswith((".jpg", ".png", ".bmp")):
            continue
        score = predict(interpreter, os.path.join(good_path, f))
        if score is not None:
            y_true.append(1)
            scores.append(score)

    duration = time.time() - start_time

    y_true = np.array(y_true)
    scores = np.array(scores, dtype=np.float32)

    if args.auto_threshold and len(scores) > 0:
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = args.threshold
        best_score = -1.0

        for t in thresholds:
            y_pred_tmp = np.where(scores > t, 0, 1)
            if args.threshold_metric == "f1":
                report = classification_report(
                    y_true,
                    y_pred_tmp,
                    target_names=["Bad", "Good"],
                    output_dict=True,
                    zero_division=0,
                )
                metric_score = report["macro avg"]["f1-score"]
            else:
                y_bad = (y_true == 0).astype(int)
                fpr, tpr, _ = roc_curve(y_bad, scores)
                metric_score = float(np.max(tpr - fpr))

            if metric_score > best_score:
                best_score = metric_score
                best_threshold = t

        args.threshold = float(best_threshold)

    y_pred = np.where(scores > args.threshold, 0, 1)

    print("\n" + "=" * 40)
    print(f"Processed {len(y_true)} images in {duration:.2f}s")
    print(f"Speed: {len(y_true)/duration:.1f} FPS")
    print(f"Threshold: {args.threshold:.2f}")
    print("=" * 40)

    target_names = ["Bad", "Good"]
    print(classification_report(y_true, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    if len(np.unique(y_true)) > 1:
        y_bad = (y_true == 0).astype(int)
        try:
            roc_auc = roc_auc_score(y_bad, scores)
            fpr, tpr, _ = roc_curve(y_bad, scores)
            pr_precision, pr_recall, _ = precision_recall_curve(y_bad, scores)
            pr_auc = auc(pr_recall, pr_precision)
            print(f"ROC AUC (Bad as positive): {roc_auc:.4f}")
            print(f"PR AUC (Bad as positive): {pr_auc:.4f}")
        except ValueError:
            pass
    print("=" * 40)


if __name__ == "__main__":
    main()
