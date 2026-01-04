"""Integration test that exercises the full pipeline with real AAPL data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_forecast import ForecastConfig, run_pipeline


def test_run_pipeline_with_aapl_data(tmp_path):
    """Download two years of AAPL prices, train, and produce a 20-day forecast."""

    url = "https://stooq.pl/q/d/l/?s=aapl.us&i=d"
    prices = pd.read_csv(url)
    if prices.empty:
        pytest.skip("AAPL data download returned empty dataset")

    prices = prices[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    csv_path = tmp_path / "aapl.csv"
    prices.to_csv(csv_path, index=False)

    config = ForecastConfig(
        data_path=csv_path,
        lookback=60,
        horizon=20,
        n_estimators=120,
    )

    model, metrics, forecast_df = run_pipeline(config)

    assert len(forecast_df) == config.horizon
    assert forecast_df["predicted_close"].notna().all()
    assert forecast_df["date"].is_monotonic_increasing
    assert model is not None
    for value in metrics.values():
        assert np.isfinite(value)
