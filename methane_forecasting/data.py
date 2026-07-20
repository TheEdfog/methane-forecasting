from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

DEFAULT_DATASET_PATH = Path("data/filtered_data_frame.csv")
DEFAULT_GOOGLE_DRIVE_FILE_ID = "1hz0dj5TVr0fFPI-mqQf7-s3JkiqQYmrx"


DatasetDownloader = Callable[[str, Path], Path]


def download_from_google_drive(file_id: str, destination: Path) -> Path:
    """Download a public Google Drive file, keeping partial data for resume."""
    import gdown

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(f"{destination.suffix}.part")
    downloaded = gdown.download(
        id=file_id,
        output=str(partial),
        quiet=False,
        resume=True,
    )
    if downloaded is None or not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(
            "Google Drive did not return the dataset. "
            "Check that the source file is still public, then retry the command."
        )
    partial.replace(destination)
    return destination


def ensure_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
    *,
    file_id: str = DEFAULT_GOOGLE_DRIVE_FILE_ID,
    download_if_missing: bool = True,
    downloader: DatasetDownloader | None = None,
) -> Path:
    """Return the local dataset, downloading the original file when requested."""
    dataset = Path(path)
    if dataset.is_file():
        return dataset
    if not download_if_missing:
        raise FileNotFoundError(
            f"dataset not found: {dataset}. Place the CSV there or allow automatic download."
        )

    fetch = downloader or download_from_google_drive
    return fetch(file_id, dataset)
