import os
import sys
import time
import urllib.request
from pathlib import Path
from zipfile import ZipFile

from maskpass import askpass


class DownloadProgress:

    def __init__(self):
        self.start_time: float | None = None

    def __call__(self, count: int, block_size: int, total_size: int):
        if count == 0:
            self.start_time = time.time()
            return

        assert self.start_time is not None
        duration = time.time() - self.start_time
        progress_size = count * block_size
        speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
        percent = int(count * block_size * 100 / total_size) if total_size else 0

        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024 * 1024), speed, int(duration))
        )
        sys.stdout.flush()


def _is_valid_zip(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        with ZipFile(file_path) as zf:
            zf.testzip()
        return True
    except Exception:
        return False


def download_datasets(repo_root: Path) -> None:
    dataset_dir = repo_root / "dataset"
    certh_dir = dataset_dir / "CERTH"
    live_dir = dataset_dir / "LIVE"

    certh_dir.mkdir(parents=True, exist_ok=True)
    live_dir.mkdir(parents=True, exist_ok=True)

    certh_zip = certh_dir / "CERTH_ImageBlurDataset.zip"

    if certh_zip.exists() and _is_valid_zip(certh_zip):
        print("CERTH zip already exists. Skipping download.")
    else:
        if certh_zip.exists():
            print("Found corrupted CERTH zip. Deleting and re-downloading...")
            certh_zip.unlink(missing_ok=True)

        print("Downloading CERTH Image Blur Dataset...")
        progress = DownloadProgress()
        urllib.request.urlretrieve(
            "https://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip",
            str(certh_zip),
            progress,
        )
        print()

    print(
        "To download the LIVE Image Quality Database, visit: "
        "http://live.ece.utexas.edu/research/quality/subjective.htm and "
        "place 'databaserelease2.zip' in: ./dataset/LIVE/"
    )

    # Extract LIVE (password-protected)
    live_zip = live_dir / "databaserelease2.zip"
    live_out = live_dir / "databaserelease2"
    if live_zip.exists() and not live_out.exists():
        password = askpass("Enter password for LIVE dataset zip: ")
        with ZipFile(live_zip) as zf:
            zf.extractall(path=live_dir, pwd=password.encode("utf-8"))

    # Extract CERTH
    certh_out = certh_dir / "CERTH_ImageBlurDataset"
    if certh_zip.exists() and not certh_out.exists():
        if not _is_valid_zip(certh_zip):
            raise ValueError(f"Invalid zip file: {certh_zip}")
        with ZipFile(certh_zip) as zf:
            zf.extractall(path=certh_dir)

    print("Done.")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    download_datasets(repo)


if __name__ == "__main__":
    main()
