from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def build_supervised_frame(
    frame: pd.DataFrame,
    *,
    target_column: str = "MM263",
    timestamp_column: str = "timestamp",
    horizon_steps: int = 900,
    lags: Sequence[int] = (1, 30, 120, 300),
    rolling_windows: Sequence[int] = (30, 120, 300),
) -> tuple[pd.DataFrame, pd.Series]:
    """Build past-only features and a future methane target."""
    if target_column not in frame:
        raise ValueError(f"missing target column: {target_column}")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive")

    ordered = frame.copy()
    if timestamp_column in ordered:
        ordered[timestamp_column] = pd.to_datetime(ordered[timestamp_column], errors="raise")
        ordered = ordered.sort_values(timestamp_column).reset_index(drop=True)

    numeric_columns = ordered.select_dtypes(include="number").columns.tolist()
    if target_column not in numeric_columns:
        raise ValueError(f"target column must be numeric: {target_column}")

    features = ordered[numeric_columns].astype(float).copy()
    for lag in sorted(set(lags)):
        if lag <= 0:
            raise ValueError("lags must be positive")
        for column in numeric_columns:
            features[f"{column}_lag_{lag}"] = ordered[column].shift(lag)

    for window in sorted(set(rolling_windows)):
        if window <= 1:
            raise ValueError("rolling windows must be greater than one")
        history = ordered[target_column].shift(1).rolling(window=window, min_periods=window)
        features[f"{target_column}_mean_{window}"] = history.mean()
        features[f"{target_column}_std_{window}"] = history.std()

    if timestamp_column in ordered:
        elapsed = (ordered[timestamp_column] - ordered[timestamp_column].iloc[0]).dt.total_seconds()
        features["hour_sin"] = np.sin(2 * np.pi * elapsed / 3_600)
        features["hour_cos"] = np.cos(2 * np.pi * elapsed / 3_600)

    target = ordered[target_column].shift(-horizon_steps).rename("target")
    valid = features.notna().all(axis=1) & target.notna()
    return features.loc[valid].reset_index(drop=True), target.loc[valid].reset_index(drop=True)


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split without shuffling so future observations never enter training."""
    if len(features) != len(target):
        raise ValueError("features and target must have the same length")
    if not 0.5 <= train_fraction < 1:
        raise ValueError("train_fraction must be in [0.5, 1)")

    split_at = int(len(features) * train_fraction)
    if split_at == 0 or split_at == len(features):
        raise ValueError("not enough rows for a chronological split")
    return (
        features.iloc[:split_at].copy(),
        features.iloc[split_at:].copy(),
        target.iloc[:split_at].copy(),
        target.iloc[split_at:].copy(),
    )
