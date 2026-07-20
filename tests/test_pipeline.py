import numpy as np
import pandas as pd
import pytest

from methane_forecasting.data import ensure_dataset
from methane_forecasting.features import build_supervised_frame, chronological_split
from methane_forecasting.model import train_forecaster


def generate_sensor_data(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seconds = np.arange(rows, dtype=float)
    methane = 0.42 + 0.11 * np.sin(2 * np.pi * seconds / 3_600)
    methane += 0.045 * np.sin(2 * np.pi * seconds / 420 + 0.8)
    methane += rng.normal(0, 0.012, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="s"),
            "MM263": methane,
            "MM264": methane * 0.84 + rng.normal(0, 0.018, rows),
            "airflow": 4.8 - methane * 1.7 + rng.normal(0, 0.08, rows),
        }
    )


def test_target_is_shifted_into_the_future():
    frame = generate_sensor_data(rows=500, seed=1)
    features, target = build_supervised_frame(
        frame,
        horizon_steps=12,
        lags=(1, 5),
        rolling_windows=(10,),
    )
    first_original_index = 10
    assert np.isclose(target.iloc[0], frame.loc[first_original_index + 12, "MM263"])
    assert np.isclose(features.iloc[0]["MM263_lag_1"], frame.loc[first_original_index - 1, "MM263"])


def test_time_features_represent_time_of_day() -> None:
    frame = generate_sensor_data(rows=200, seed=7)
    frame.loc[:, "timestamp"] = pd.date_range("2025-01-01 06:00:00", periods=200, freq="24h")
    features, _ = build_supervised_frame(
        frame,
        horizon_steps=1,
        lags=(1,),
        rolling_windows=(2,),
    )
    assert features["time_of_day_sin"].nunique() == 1
    assert features["time_of_day_cos"].nunique() == 1


def test_split_is_chronological():
    frame = generate_sensor_data(rows=500, seed=2)
    features, target = build_supervised_frame(
        frame,
        horizon_steps=10,
        lags=(1,),
        rolling_windows=(5,),
    )
    x_train, x_test, y_train, y_test = chronological_split(features, target, train_fraction=0.75)
    assert len(x_train) + len(x_test) == len(features)
    assert x_train.index.max() < x_test.index.min()
    assert y_train.index.max() < y_test.index.min()


def test_model_returns_finite_metrics():
    frame = generate_sensor_data(rows=2_000, seed=3)
    features, target = build_supervised_frame(
        frame,
        horizon_steps=60,
        lags=(1, 10, 30),
        rolling_windows=(20, 60),
    )
    model, report = train_forecaster(features, target)
    assert model.n_iter_ > 0
    assert np.isfinite(report.mae)
    assert np.isfinite(report.rmse)
    assert np.isfinite(report.r2)
    assert report.train_rows > report.test_rows > 0


def test_existing_dataset_is_not_downloaded(tmp_path):
    dataset = tmp_path / "sensor.csv"
    dataset.write_text("MM263\n0.1\n", encoding="utf-8")

    def fail_if_called(file_id, destination):
        raise AssertionError("downloader should not be called")

    assert ensure_dataset(dataset, downloader=fail_if_called) == dataset


def test_missing_dataset_can_be_downloaded(tmp_path):
    dataset = tmp_path / "sensor.csv"

    def fake_download(file_id, destination):
        assert file_id == "public-file"
        destination.write_text("MM263\n0.1\n", encoding="utf-8")
        return destination

    result = ensure_dataset(dataset, file_id="public-file", downloader=fake_download)
    assert result == dataset
    assert result.is_file()


def test_missing_dataset_can_fail_without_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset not found"):
        ensure_dataset(tmp_path / "missing.csv", download_if_missing=False)
