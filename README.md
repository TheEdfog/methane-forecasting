# Methane forecasting

This project predicts methane concentration from a sequence of industrial sensor readings. The forecast horizon is configurable; the original experiment used a 15 minute horizon with one reading per second.

I first built this as a university ML assignment. The old notebook depended on a private Google Drive file and used a random train-test split. That split was not appropriate for a time series because later observations could leak into training. This version uses a chronological split and keeps feature calculations limited to information available at prediction time.

## Approach

The pipeline performs four steps:

1. Sort and validate sensor readings.
2. Build lag, rolling and time-of-day features.
3. Move the methane target forward by the requested horizon.
4. Train a gradient boosting regressor and compare it with a persistence baseline.

The persistence baseline predicts that the future methane value will equal the latest reading. It is a useful reference for a slowly changing signal and prevents a complex model from looking good only because the series is smooth.

## Run the example

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m methane_forecasting generate-sample
python -m methane_forecasting train
```

The training command writes the fitted model and a JSON report to `artifacts/`. Generated artifacts and private sensor data are ignored by Git.

The repository includes a synthetic dataset generator so the full workflow can be run without access to the original industrial data. Synthetic data is only a reproducibility aid; it is not presented as measured mine data.

On the bundled synthetic sample, the checked-in configuration produced MAE `0.043` and R2 `0.618`. The persistence baseline MAE was `0.108`. These numbers verify the pipeline, but they are not evidence of performance on a real mine.

## Use another dataset

The input must contain a numeric `MM263` target column. A `timestamp` column is recommended. Other numeric columns are treated as sensor features.

```bash
python -m methane_forecasting train \
  --input path/to/readings.csv \
  --horizon 900 \
  --model-output artifacts/model.joblib \
  --report-output artifacts/metrics.json
```

## Checks

```bash
ruff check .
ruff format --check .
pytest
```

Tests cover future target construction, chronological splitting and a complete model fit on generated data.

## Repository layout

```text
methane_forecasting/   reusable data, feature and model code
notebooks/             short walkthrough by Alim Narmamatov
tests/                 leakage and pipeline checks
data/                  generated sample data
```

## Limits

The example is an offline experiment, not a mine safety system. A production deployment would need sensor calibration checks, missing-data handling, drift monitoring, alert thresholds agreed with domain experts and validation on data from the target site.
