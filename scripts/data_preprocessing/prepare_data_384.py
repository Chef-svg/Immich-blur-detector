import os
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit
from concurrent.futures import ThreadPoolExecutor
import albumentations as A

CONFIG = {
    "PATCH_SIZE": 384,
    "SPLIT_RATIOS": (0.8, 0.1, 0.1),
    "SAMPLES_PER_IMAGE": 4,
    "SEED": 42,
    "NUM_THREADS": 8,
    "MIN_VARIANCE": 50,
    "LIVE_BAD_DISTORTIONS": ["gblur", "fastfading", "jpeg", "jp2k", "wn"],
}

BASE_DIR = Path(__file__).parent.parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "prepared/data_formatted_native_384"

random.seed(CONFIG["SEED"])
np.random.seed(CONFIG["SEED"])


def get_aug(is_bad=False):
    if is_bad:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
            ]
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=15, p=0.3),
            ]
        )


def get_laplacian_var(image_array: np.ndarray) -> float:
    return float(cv2.Laplacian(image_array, cv2.CV_64F).var())


def get_all_files():
    data = []

    certh_root = DATASET_DIR / "CERTH" / "CERTH_ImageBlurDataset" / "TrainingSet"
    if certh_root.exists():
        for img in (certh_root / "Undistorted").glob("*.*"):
            if img.suffix.lower() in [".jpg", ".png", ".bmp"]:
                data.append(
                    {"path": img, "label": "good", "group": f"certh_{img.stem}"}
                )
        for folder in ["Artificially-Blurred", "Naturally-Blurred", "NewDigitalBlur"]:
            for img in (certh_root / folder).glob("*.*"):
                if img.suffix.lower() in [".jpg", ".png", ".bmp"]:
                    data.append(
                        {
                            "path": img,
                            "label": "bad",
                            "group": f"certh_{img.stem}",
                        }
                    )
    live_root = DATASET_DIR / "LIVE" / "databaserelease2"
    if live_root.exists():
        for img in (live_root / "refimgs").glob("*.*"):
            if img.name.lower() != "thumbs.db":
                data.append({"path": img, "label": "good", "group": f"live_{img.stem}"})
        for folder in CONFIG["LIVE_BAD_DISTORTIONS"]:
            for img in (live_root / folder).glob("*.bmp"):
                data.append({"path": img, "label": "bad", "group": f"live_{img.stem}"})

    return data


def process_image(args):
    file_info, split, output_root = args
    path = file_info["path"]
    label = file_info["label"]

    try:
        with open(path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w, _ = img.shape
        patch_size = CONFIG["PATCH_SIZE"]

        if h < patch_size or w < patch_size:
            return

        save_dir = output_root / split / label
        os.makedirs(save_dir, exist_ok=True)

        augmentor = get_aug(label == "bad")

        for i in range(CONFIG["SAMPLES_PER_IMAGE"]):
            top = np.random.randint(0, h - patch_size + 1)
            left = np.random.randint(0, w - patch_size + 1)

            patch = img[top : top + patch_size, left : left + patch_size]
            if label == "good" and get_laplacian_var(patch) < CONFIG["MIN_VARIANCE"]:
                continue

            augmented = augmentor(image=patch)["image"]

            out_name = f"{path.stem}_p{i}_{split}.jpg"
            save_path = save_dir / out_name

            success = False
            try:
                success = cv2.imwrite(
                    str(save_path), cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                )
            except Exception:
                success = False

            if not success:
                clean_name = f"{hash(str(path))}_p{i}_{split}.jpg"
                cv2.imwrite(
                    str(save_dir / clean_name),
                    cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR),
                )

    except Exception as e:
        print(f"Error processing {path.name}: {e}")


def main():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    all_data = get_all_files()
    print(f"Found {len(all_data)} source images.")

    groups = [x["group"] for x in all_data]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss.split(all_data, groups=groups))

    train_data = [all_data[i] for i in train_idx]
    temp_data = [all_data[i] for i in temp_idx]
    temp_groups = [groups[i] for i in temp_idx]

    gss_val_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val_test.split(temp_data, groups=temp_groups))

    val_data = [temp_data[i] for i in val_idx]
    test_data = [temp_data[i] for i in test_idx]

    datasets = [(train_data, "train"), (val_data, "val"), (test_data, "test")]

    tasks = []
    for data_list, split_name in datasets:
        for item in data_list:
            tasks.append((item, split_name, OUTPUT_DIR))

    print(f"Processing {len(tasks)} tasks with {CONFIG['NUM_THREADS']} threads...")
    print(f"Target Resolution: {CONFIG['PATCH_SIZE']}x{CONFIG['PATCH_SIZE']}")

    with ThreadPoolExecutor(max_workers=CONFIG["NUM_THREADS"]) as ex:
        list(tqdm(ex.map(process_image, tasks), total=len(tasks)))


if __name__ == "__main__":
    main()
