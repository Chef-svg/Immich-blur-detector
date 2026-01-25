import os
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import regularizers
import keras
from pathlib import Path


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
IMG_SIZE = 384
BATCH_SIZE = 16
MOBILENET_ALPHA = 1.4
HEAD_UNITS = 768
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "prepared/data_formatted_native_384"
MODEL_NAME = BASE_DIR / "models/data_formatted_native_384/quality_model_max"
TFLITE_INT8_PATH = f"{MODEL_NAME}_int8.tflite"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU Acceleration Enabled: {len(gpus)} device(s)")


def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mix_up(ds_one, ds_two, alpha=0.2):
    images_one, labels_one = ds_one
    images_two, labels_two = ds_two
    batch_size = tf.shape(images_one)[0]

    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    images = images_one * x_l + images_two * (1 - x_l)
    labels = labels_one * y_l + labels_two * (1 - y_l)
    return (images, labels)


print(f"Loading Data from {DATA_DIR}...")

train_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
)
val_ds_raw = keras.utils.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image, label


def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, label


AUTOTUNE = tf.data.AUTOTUNE


def build_train_stream(seed: int):
    stream = train_ds_raw.unbatch()
    stream = stream.map(preprocess, num_parallel_calls=AUTOTUNE)
    stream = stream.map(augment, num_parallel_calls=AUTOTUNE)
    stream = stream.shuffle(2000, seed=seed, reshuffle_each_iteration=True)
    return stream


ds_one = build_train_stream(42).batch(BATCH_SIZE, drop_remainder=True)
ds_two = build_train_stream(43).batch(BATCH_SIZE, drop_remainder=True)

train_ds = tf.data.Dataset.zip((ds_one, ds_two))
train_ds = train_ds.map(mix_up, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

val_ds = (
    val_ds_raw.map(preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
)


def build_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    imagenet_allowed = {0.35, 0.50, 0.75, 1.0, 1.3, 1.4}
    if MOBILENET_ALPHA not in imagenet_allowed:
        raise ValueError(
            "MobileNetV2 ImageNet weights only support alpha in "
            f"{sorted(imagenet_allowed)}; got alpha={MOBILENET_ALPHA}. "
            "Either pick a supported alpha or set weights=None to train from scratch."
        )

    base = MobileNetV2(
        input_tensor=inputs,
        alpha=MOBILENET_ALPHA,
        include_top=False,
        weights="imagenet",
    )

    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(HEAD_UNITS, activation="relu", kernel_regularizer=regularizers.l2(0.01))(
        x
    )
    x = Dropout(0.3)(x)

    outputs = Dense(2, activation="softmax")(x)

    return Model(inputs, outputs)


def get_class_weights(train_dir: Path):
    class_counts = {}
    for class_dir in train_dir.iterdir():
        if not class_dir.is_dir():
            continue
        count = len(
            [
                p
                for p in class_dir.glob("**/*")
                if p.suffix.lower() in [".jpg", ".png", ".bmp"]
            ]
        )
        class_counts[class_dir.name] = count

    total = sum(class_counts.values())
    if total == 0:
        return None

    class_names = sorted(class_counts.keys())
    weights = {}
    for idx, name in enumerate(class_names):
        weights[idx] = total / max(1, len(class_names) * class_counts[name])
    return weights


model = build_model()
print(
    f"🏗️ Model Built: MobileNetV2 ({MOBILENET_ALPHA}x) @ {IMG_SIZE}x{IMG_SIZE} with head {HEAD_UNITS}"
)
class_weights = get_class_weights(DATA_DIR / "train")

print("\nSTAGE 1: Training Head...")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

model.fit(train_ds, validation_data=val_ds, epochs=6, class_weight=class_weights)

print("\nSTAGE 2: Fine Tuning...")
model.trainable = True

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint(
        f"{MODEL_NAME}_best.h5", save_best_only=True, monitor="val_accuracy", mode="max"
    ),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.2, min_lr=1e-7),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks,
    class_weight=class_weights,
)

print("\nApplying QAT and Exporting...")

model.load_weights(f"{MODEL_NAME}_best.h5")

quantize_model = tfmot.quantization.keras.quantize_model
qat_model = quantize_model(model)

qat_model.compile(
    optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"]
)
qat_model.fit(train_ds, validation_data=val_ds, epochs=2)

converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]


def representative_data_gen():
    ds_calib = train_ds_raw.unbatch().map(preprocess).batch(1).take(100)
    for images, _ in ds_calib:
        yield [images]


converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

try:
    tflite_model = converter.convert()
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nDONE! Saved: {TFLITE_INT8_PATH}")
    print(f"Model size (int8): {size_mb:.2f} MB")
    print("\nNext step: run the EdgeTPU compiler to generate a compiled model, e.g.:")
    print(f"  edgetpu_compiler -o {MODEL_NAME.parent} {TFLITE_INT8_PATH}")
except Exception as e:
    print(f"Export Error: {e}")
