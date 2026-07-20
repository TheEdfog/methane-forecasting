from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from methane_forecasting.features import chronological_split


@dataclass(frozen=True)
class ForecastReport:
    mae: float
    rmse: float
    r2: float
    persistence_mae: float
    train_rows: int
    test_rows: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def evaluate_forecast(
    actual: pd.Series | np.ndarray,
    predicted: np.ndarray,
    persistence: pd.Series | np.ndarray,
    *,
    train_rows: int = 0,
) -> ForecastReport:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    persistence_values = np.asarray(persistence, dtype=float)
    return ForecastReport(
        mae=float(mean_absolute_error(actual_values, predicted_values)),
        rmse=float(mean_squared_error(actual_values, predicted_values) ** 0.5),
        r2=float(r2_score(actual_values, predicted_values)),
        persistence_mae=float(mean_absolute_error(actual_values, persistence_values)),
        train_rows=train_rows,
        test_rows=len(actual_values),
    )


def train_forecaster(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_column: str = "MM263",
    train_fraction: float = 0.7,
    random_state: int = 42,
    model_path: str | Path | None = None,
) -> tuple[HistGradientBoostingRegressor, ForecastReport]:
    x_train, x_test, y_train, y_test = chronological_split(
        features,
        target,
        train_fraction=train_fraction,
    )
    if target_column not in x_test:
        raise ValueError(f"missing persistence feature: {target_column}")

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=180,
        max_leaf_nodes=24,
        l2_regularization=0.1,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    report = evaluate_forecast(
        y_test,
        predictions,
        x_test[target_column],
        train_rows=len(x_train),
    )

    if model_path is not None:
        destination = Path(model_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": list(features.columns)}, destination)
    return model, report
