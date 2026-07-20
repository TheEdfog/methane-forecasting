# Methane forecasting

This project predicts methane concentration from a sequence of industrial sensor readings. The forecast horizon is configurable; the original experiment used a 15 minute horizon with one reading per second.

I first built this as a university ML assignment. The old notebook depended on Google Drive and used a random train-test split. That split was not appropriate for a time series because later observations could leak into training. This version uses a chronological split and keeps feature calculations limited to information available at prediction time.

## Approach

The pipeline performs four steps:

1. Sort and validate sensor readings.
2. Build lag, past-only rolling and cyclical time-of-day features.
3. Move the methane target forward by the requested horizon.
4. Train a gradient boosting regressor and compare it with a persistence baseline.

The persistence baseline predicts that the future methane value will equal the latest reading. It is a useful reference for a slowly changing signal and prevents a complex model from looking good only because the series is smooth.

## Run the example

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m methane_forecasting train
```

The training command writes the fitted model and a JSON report to `artifacts/`. Generated artifacts and private sensor data are ignored by Git.

By default the command reads `data/filtered_data_frame.csv`. If that file is absent, it downloads the [original public file from Google Drive](https://drive.google.com/file/d/1hz0dj5TVr0fFPI-mqQf7-s3JkiqQYmrx/view). Downloads are resumable and go through a `.part` file, so an interrupted transfer does not replace a valid dataset.

The dataset is large and is not stored in Git. To download it without starting training:

```bash
python -m methane_forecasting download-data
```

To use a file that was copied into the folder manually, keep the default name or pass its path with `--input`. Add `--no-download` if a missing local file should be treated as an error.

## Use another dataset

The input must contain a numeric `MM263` target column. A `timestamp` column is recommended. Other numeric columns are treated as sensor features.

```bash
python -m methane_forecasting train \
  --input data/another_sensor_export.csv \
  --no-download \
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

Tests cover dataset resolution without network access, future target construction, chronological splitting and a complete model fit on generated test data.

## Repository layout

```text
methane_forecasting/   reusable data, feature and model code
notebooks/             short walkthrough by Alim Narmamatov
tests/                 leakage and pipeline checks
data/                  local dataset folder; CSV files are ignored by Git
```

## Limits

The example is an offline experiment, not a mine safety system. A production deployment would need sensor calibration checks, missing-data handling, drift monitoring, alert thresholds agreed with domain experts and validation on data from the target site.
