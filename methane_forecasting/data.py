from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_data(
    rows: int = 12_000,
    *,
    seed: int = 42,
    frequency_seconds: int = 1,
) -> pd.DataFrame:
    """Generate a small, realistic sensor dataset for examples and tests."""
    if rows < 200:
        raise ValueError("rows must be at least 200")
    if frequency_seconds <= 0:
        raise ValueError("frequency_seconds must be positive")

    rng = np.random.default_rng(seed)
    seconds = np.arange(rows, dtype=float) * frequency_seconds
    slow_cycle = 0.11 * np.sin(2 * np.pi * seconds / 3_600)
    ventilation_cycle = 0.045 * np.sin(2 * np.pi * seconds / 420 + 0.8)
    drift = 0.000004 * seconds
    noise = rng.normal(0, 0.012, rows)

    methane = 0.42 + slow_cycle + ventilation_cycle + drift + noise
    margin = min(300, max(25, rows // 10))
    possible_centres = np.arange(margin, rows - margin)
    event_count = min(max(1, rows // 3_000), len(possible_centres))
    event_centres = rng.choice(possible_centres, size=event_count, replace=False)
    for centre in event_centres:
        width = rng.integers(45, 150)
        amplitude = rng.uniform(0.08, 0.22)
        methane += amplitude * np.exp(-0.5 * ((np.arange(rows) - centre) / width) ** 2)

    methane = np.clip(methane, 0, 1.4)
    timestamp = pd.date_range("2025-01-01", periods=rows, freq=f"{frequency_seconds}s")

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "MM263": methane,
            "MM264": np.clip(methane * 0.84 + rng.normal(0, 0.018, rows), 0, None),
            "airflow": 4.8 - methane * 1.7 + rng.normal(0, 0.08, rows),
            "pressure": 101.4 + 0.18 * np.sin(seconds / 700) + rng.normal(0, 0.03, rows),
            "temperature": 18.5 + 0.7 * np.sin(seconds / 2_400) + rng.normal(0, 0.08, rows),
        }
    )


def write_sample(path: str | Path, rows: int = 12_000, *, seed: int = 42) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    generate_sensor_data(rows=rows, seed=seed).to_csv(destination, index=False)
    return destination
