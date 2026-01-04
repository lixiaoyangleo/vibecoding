"""Unit tests for the walk-forward backtest utility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stock_forecast import (
    add_indicators,
    default_feature_columns,
    walk_forward_backtest,
)


def test_walk_forward_backtest_returns_metrics():
    dates = pd.date_range("2020-01-01", periods=160, freq="D")
    trend = np.linspace(100, 120, len(dates))
    prices = trend + 2 * np.sin(np.linspace(0, 10, len(dates)))
    df = pd.DataFrame({"date": dates, "close": prices})
    enriched = add_indicators(df, "close")

    results = walk_forward_backtest(
        enriched,
        feature_cols=default_feature_columns("close"),
        target_col="close",
        lookback=30,
        horizon=5,
        n_estimators=20,
        random_state=0,
        step=10,
    )

    assert not results.empty
    assert set(["cutoff_date", "mae", "rmse", "r2"]).issubset(results.columns)
    assert (results["mae"] >= 0).all()
