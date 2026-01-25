import tensorflow as tf
import numpy as np
import os
import datetime
import tensorflow.keras as keras
from sklearn.metrics import classification_report, confusion_matrix


def generate_report(
    model_path, data_dir, image_size=(224, 224), report_file="report.txt"
):
    print(f"\nStarting Diagnostic on: {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("   Loading validation data...")
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "val"),
        image_size=image_size,
        batch_size=32,
        shuffle=False,
        label_mode="categorical",
    )

    class_names = val_ds.class_names
    file_paths = val_ds.file_paths

    def preprocess_grayscale(image, label):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image, label

    val_ds_processed = val_ds.map(preprocess_grayscale)

    print("   Running inference...")
    y_pred_probs = model.predict(val_ds_processed, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = []
    for images, labels in val_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_true = np.array(y_true)

    errors = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            filename = os.path.basename(file_paths[i])
            true_label = class_names[y_true[i]]
            pred_label = class_names[y_pred[i]]
            confidence = y_pred_probs[i][y_pred[i]]
            errors.append(
                f"File: {filename} | True: {true_label} | Pred: {pred_label} ({confidence:.2f})"
            )

    with open(report_file, "w") as f:
        f.write(f"=== DIAGNOSTIC REPORT ===\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Date: {datetime.datetime.now()}\n\n")

        f.write("--- CONFUSION MATRIX ---\n")
        cm = confusion_matrix(y_true, y_pred)
        f.write(f"Classes: {class_names}\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("--- CLASSIFICATION REPORT ---\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
        f.write("\n\n")

        f.write(f"--- MISCLASSIFIED IMAGES ({len(errors)}) ---\n")
        f.write("Images the model got wrong (Confidence in brackets):\n")
        for err in errors[:50]:
            f.write(err + "\n")
        if len(errors) > 50:
            f.write(f"... and {len(errors)-50} more.\n")

    print(f"Report generated: {report_file}")
