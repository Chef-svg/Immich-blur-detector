import argparse
import io
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_DIR = BASE_DIR / "models" / "data_formatted_native_384"
DEFAULT_STATE_DB = BASE_DIR / "immich_tagger_state.db"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tag blurry images in Immich using EdgeTPU"
    )
    parser.add_argument(
        "--immich-url",
        default=os.getenv("IMMICH_URL", "http://localhost:2283"),
        help="Immich server URL (default: from IMMICH_URL env or http://localhost:2283)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("IMMICH_API_KEY"),
        help="Immich API key (default: from IMMICH_API_KEY env)",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH"),
        help="Path to TFLite model (auto-detects if omitted)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("BLUR_THRESHOLD", "0.5")),
        help="Blur score threshold (default: 0.5)",
    )
    parser.add_argument(
        "--tag-name",
        default=os.getenv("BLUR_TAG_NAME", "blur:bad"),
        help="Tag name for blurry images",
    )
    parser.add_argument(
        "--good-tag-name",
        default=os.getenv("GOOD_TAG_NAME", "blur:good"),
        help="Tag name for sharp/good images",
    )
    parser.add_argument(
        "--state-db", default=str(DEFAULT_STATE_DB), help="Path to database"
    )
    parser.add_argument("--force", action="store_true", help="Reprocess all assets")
    parser.add_argument(
        "--once", action="store_true", help="Process once and exit (no polling)"
    )
    parser.add_argument(
        "--poll-interval", type=int, default=300, help="Poll interval in seconds"
    )
    parser.add_argument("--page-size", type=int, default=100, help="Assets per page")
    parser.add_argument(
        "--dry-run", action="store_true", help="Don't actually tag, just print"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("DOWNLOAD_WORKERS", "8")),
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("API key required: set IMMICH_API_KEY in .env or use --api-key")

    return args


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

    raise FileNotFoundError("No suitable TFLite model found in " + str(MODEL_DIR))


def load_interpreter(model_path: Path, use_edgetpu: bool):
    if use_edgetpu:
        try:
            from pycoral.utils import edgetpu

            interpreter = edgetpu.make_interpreter(str(model_path))
            print(f"[TPU] Loaded EdgeTPU model: {model_path.name}")
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            print(f"[WARNING] EdgeTPU failed: {e}")
            print("[WARNING] Falling back to CPU mode...")
            cpu_model = model_path.parent / model_path.name.replace("_edgetpu", "")
            if cpu_model.exists() and cpu_model != model_path:
                model_path = cpu_model
                print(f"[WARNING] Using CPU model: {model_path.name}")

    try:
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(model_path))

    print(f"[CPU] Loaded CPU model: {model_path.name}")
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(img: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
    img = cv2.resize(img, input_shape)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    return img


def quantize_input(img: np.ndarray, input_details) -> np.ndarray:
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


def dequantize_output(output: np.ndarray, output_details) -> np.ndarray:
    dtype = output_details[0]["dtype"]
    scale, zero_point = output_details[0]["quantization"]
    if dtype in (np.uint8, np.int8):
        return (output.astype(np.float32) - zero_point) * scale
    return output.astype(np.float32)


def predict(interpreter, img_bytes: bytes) -> float:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    width, height = int(input_shape[2]), int(input_shape[1])

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_image(img, (width, height))
    img = quantize_input(img, input_details)

    input_index = input_details[0]["index"]
    interpreter.set_tensor(input_index, np.expand_dims(img, axis=0))
    interpreter.invoke()

    output_index = output_details[0]["index"]
    output = interpreter.get_tensor(output_index)[0]
    output = dequantize_output(output, output_details)
    return float(output[0]) if output.shape[0] >= 2 else float(output[0])


class ImmichClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {"x-api-key": api_key}

    def get_tags(self):
        r = requests.get(f"{self.base_url}/api/tags", headers=self.headers)
        r.raise_for_status()
        return r.json()

    def create_tag(self, name: str):
        r = requests.post(
            f"{self.base_url}/api/tags", headers=self.headers, json={"name": name}
        )
        r.raise_for_status()
        return r.json()

    def ensure_tag(self, name: str) -> str:
        for tag in self.get_tags():
            if tag.get("name") == name:
                return tag["id"]
        return self.create_tag(name)["id"]

    def add_tag_to_assets(self, tag_id: str, asset_ids: List[str]):
        if not asset_ids:
            return
        r = requests.put(
            f"{self.base_url}/api/tags/{tag_id}/assets",
            headers=self.headers,
            json={"ids": asset_ids},
        )
        r.raise_for_status()

    def search_assets(self, page: int, size: int):
        r = requests.post(
            f"{self.base_url}/api/search/metadata",
            headers=self.headers,
            json={"type": "IMAGE", "page": page, "size": size},
        )
        r.raise_for_status()
        data = r.json()
        assets_data = data.get("assets", {})
        items = assets_data.get("items", [])
        total = assets_data.get("total", 0)
        return items, total

    def download_asset(self, asset_id: str) -> bytes:
        r = requests.get(
            f"{self.base_url}/api/assets/{asset_id}/thumbnail",
            headers=self.headers,
            params={"size": "preview"},
            timeout=30,
        )
        r.raise_for_status()
        return r.content


class StateDB:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed (
                asset_id TEXT PRIMARY KEY,
                checksum TEXT,
                decision TEXT,
                processed_at TEXT
            )
        """
        )
        self.conn.commit()

    def is_processed(self, asset_id: str, checksum: str) -> bool:
        cur = self.conn.execute(
            "SELECT checksum FROM processed WHERE asset_id = ?", (asset_id,)
        )
        row = cur.fetchone()
        return row is not None and row[0] == checksum

    def mark_processed(self, asset_id: str, checksum: str, decision: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO processed (asset_id, checksum, decision, processed_at) VALUES (?, ?, ?, ?)",
            (asset_id, checksum, decision, datetime.now(timezone.utc).isoformat()),
        )
        self.conn.commit()


def download_one(client: ImmichClient, asset_id: str) -> Tuple[str, Optional[bytes]]:
    try:
        return asset_id, client.download_asset(asset_id)
    except Exception:
        return asset_id, None


def process_assets(
    args,
    interpreter,
    client: ImmichClient,
    state: StateDB,
    bad_tag_id: str,
    good_tag_id: str,
):
    page = 1
    processed = 0
    tagged_bad = 0
    tagged_good = 0
    skipped = 0
    errors = 0

    _, total = client.search_assets(1, 1)
    print(f"Found {total} total image assets")
    print(f"Using {args.workers} download workers")

    pbar = tqdm(total=total, desc="Processing", unit="img")

    while True:
        assets, _ = client.search_assets(page, args.page_size)
        if not assets:
            break

        to_process: List[Dict] = []
        for asset in assets:
            asset_id = asset.get("id")
            checksum = asset.get("checksum", "")
            if not asset_id:
                pbar.update(1)
                continue
            if not args.force and state.is_processed(asset_id, checksum):
                skipped += 1
                pbar.update(1)
                pbar.set_postfix(
                    bad=tagged_bad, good=tagged_good, skip=skipped, err=errors
                )
                continue
            to_process.append(asset)

        if not to_process:
            page += 1
            continue

        asset_map = {a["id"]: a for a in to_process}
        downloaded: Dict[str, bytes] = {}

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, client, a["id"]): a["id"]
                for a in to_process
            }
            for future in as_completed(futures):
                asset_id, img_bytes = future.result()
                if img_bytes:
                    downloaded[asset_id] = img_bytes

        batch_bad: List[str] = []
        batch_good: List[str] = []

        for asset in to_process:
            asset_id = asset["id"]
            checksum = asset.get("checksum", "")

            if asset_id not in downloaded:
                tqdm.write(f"ERROR {asset_id[:8]}...: download failed")
                state.mark_processed(asset_id, checksum, "error")
                errors += 1
                pbar.update(1)
                continue

            try:
                score = predict(interpreter, downloaded[asset_id])
                is_blurry = score >= args.threshold
                decision = "bad" if is_blurry else "good"

                if not args.dry_run:
                    if is_blurry:
                        batch_bad.append(asset_id)
                    else:
                        batch_good.append(asset_id)

                state.mark_processed(asset_id, checksum, f"{decision}:{score:.4f}")
                processed += 1

            except Exception as e:
                tqdm.write(f"ERROR {asset_id[:8]}...: {e}")
                state.mark_processed(asset_id, checksum, "error")
                errors += 1

            pbar.update(1)
            pbar.set_postfix(
                bad=tagged_bad + len(batch_bad),
                good=tagged_good + len(batch_good),
                skip=skipped,
                err=errors,
            )

        if not args.dry_run:
            try:
                client.add_tag_to_assets(bad_tag_id, batch_bad)
                tagged_bad += len(batch_bad)
            except Exception as e:
                tqdm.write(f"ERROR tagging bad batch: {e}")
            try:
                client.add_tag_to_assets(good_tag_id, batch_good)
                tagged_good += len(batch_good)
            except Exception as e:
                tqdm.write(f"ERROR tagging good batch: {e}")

        pbar.set_postfix(bad=tagged_bad, good=tagged_good, skip=skipped, err=errors)
        page += 1

    pbar.close()
    return processed, tagged_bad, tagged_good, skipped


def main():
    args = parse_args()

    print("=" * 50)
    print("Immich Blur Tagger (EdgeTPU)")
    print("=" * 50)

    model_path, use_edgetpu = resolve_model_path(args.model_path)
    interpreter = load_interpreter(model_path, use_edgetpu)

    print(f"Connecting to Immich at {args.immich_url}")
    client = ImmichClient(args.immich_url, args.api_key)

    bad_tag_id = client.ensure_tag(args.tag_name)
    good_tag_id = client.ensure_tag(args.good_tag_name)
    print(f"Using tags: '{args.tag_name}' (bad), '{args.good_tag_name}' (good)")

    state = StateDB(args.state_db)

    print(
        f"Threshold: {args.threshold}, Dry run: {args.dry_run}, Force: {args.force}, Workers: {args.workers}"
    )
    print("=" * 50)

    while True:
        start = time.time()
        processed, tagged_bad, tagged_good, skipped = process_assets(
            args, interpreter, client, state, bad_tag_id, good_tag_id
        )
        elapsed = time.time() - start

        print(
            f"\n✓ Done: {processed} processed, {tagged_bad} bad, {tagged_good} good, {skipped} skipped ({elapsed:.1f}s)"
        )

        if args.once:
            break

        print(f"Sleeping {args.poll_interval}s until next scan...")
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
