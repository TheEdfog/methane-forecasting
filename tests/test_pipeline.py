import numpy as np

from methane_forecasting.data import generate_sensor_data
from methane_forecasting.features import build_supervised_frame, chronological_split
from methane_forecasting.model import train_forecaster


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
