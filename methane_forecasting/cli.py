from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from methane_forecasting.data import write_sample
from methane_forecasting.features import build_supervised_frame
from methane_forecasting.model import train_forecaster


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a methane sensor forecasting model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-sample", help="write a synthetic sensor CSV")
    generate.add_argument("--output", default="data/sample_methane.csv")
    generate.add_argument("--rows", type=int, default=12_000)

    train = subparsers.add_parser("train", help="train and evaluate a forecasting model")
    train.add_argument("--input", default="data/sample_methane.csv")
    train.add_argument("--horizon", type=int, default=900, help="forecast horizon in rows")
    train.add_argument("--model-output", default="artifacts/methane_forecaster.joblib")
    train.add_argument("--report-output", default="artifacts/metrics.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate-sample":
        destination = write_sample(args.output, rows=args.rows)
        print(destination)
        return 0

    frame = pd.read_csv(args.input)
    features, target = build_supervised_frame(frame, horizon_steps=args.horizon)
    _, report = train_forecaster(features, target, model_path=args.model_output)
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0
