from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from methane_forecasting.data import (
    DEFAULT_DATASET_PATH,
    DEFAULT_GOOGLE_DRIVE_FILE_ID,
    ensure_dataset,
)
from methane_forecasting.features import build_supervised_frame
from methane_forecasting.model import train_forecaster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a methane sensor forecasting model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="download the original sensor CSV")
    download.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    download.add_argument("--file-id", default=DEFAULT_GOOGLE_DRIVE_FILE_ID)

    train = subparsers.add_parser("train", help="train and evaluate a forecasting model")
    train.add_argument("--input", default=str(DEFAULT_DATASET_PATH))
    train.add_argument("--file-id", default=DEFAULT_GOOGLE_DRIVE_FILE_ID)
    train.add_argument(
        "--no-download",
        action="store_true",
        help="fail instead of downloading when the input file is missing",
    )
    train.add_argument("--horizon", type=int, default=900, help="forecast horizon in rows")
    train.add_argument("--model-output", default="artifacts/methane_forecaster.joblib")
    train.add_argument("--report-output", default="artifacts/metrics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        destination = ensure_dataset(args.output, file_id=args.file_id)
        print(destination)
        return 0

    dataset = ensure_dataset(
        args.input,
        file_id=args.file_id,
        download_if_missing=not args.no_download,
    )
    frame = pd.read_csv(dataset)
    features, target = build_supervised_frame(frame, horizon_steps=args.horizon)
    _, report = train_forecaster(features, target, model_path=args.model_output)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0
