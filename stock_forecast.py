"""
End-to-end example for training a sliding-window regression model on stock prices.

This script assumes you have at least two years of daily price data with columns:
- a date column (default: "date") that can be parsed by pandas
- a closing price column (default: "close")

It produces:
- walk-forward train/validation/test splits
- a multi-output RandomForest model predicting the next `horizon` days
- a CSV with the forecasted trajectory for the next month

Usage:
    python stock_forecast.py \
        --data your_prices.csv \
        --date-col date \
        --price-col close \
        --lookback 60 \
        --horizon 20 \
        --output forecast.csv

The code is intentionally lightweight and focuses on a reproducible baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class ForecastConfig:
    data_path: Path
    date_col: str = "date"
    price_col: str = "close"
    lookback: int = 60
    horizon: int = 20
    n_estimators: int = 400
    random_state: int = 42
    output_path: Path = Path("forecast.csv")
    backtest_output: Optional[Path] = None


def default_feature_columns(price_col: str) -> List[str]:
    return [
        price_col,
        "log_return",
        "ma_5",
        "ma_20",
        "vol_5",
        "vol_20",
    ]


def load_price_data(path: Path, date_col: str, price_col: str) -> pd.DataFrame:
    """Load and sort the price data, keeping only the needed columns."""
    df = pd.read_csv(path)
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column '{date_col}' in {path}")
    if price_col not in df.columns:
        raise ValueError(f"Missing required price column '{price_col}' in {path}")
    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def add_indicators(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Compute basic technical indicators to enrich the feature space."""
    enriched = df.copy()
    enriched["log_return"] = np.log(enriched[price_col]).diff()
    enriched["ma_5"] = enriched[price_col].rolling(5, min_periods=1).mean()
    enriched["ma_20"] = enriched[price_col].rolling(20, min_periods=1).mean()
    enriched["vol_5"] = enriched["log_return"].rolling(5, min_periods=1).std()
    enriched["vol_20"] = enriched["log_return"].rolling(20, min_periods=1).std()
    enriched.fillna(method="bfill", inplace=True)
    return enriched


def build_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Transform a univariate series with indicators into supervised samples."""
    features, targets = [], []
    values = df[feature_cols + [target_col]].values
    for idx in range(lookback, len(df) - horizon + 1):
        window = values[idx - lookback : idx]
        target = values[idx : idx + horizon, -1]
        features.append(window[:, : len(feature_cols)].flatten())
        targets.append(target)
    return np.array(features), np.array(targets)


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    n_estimators: int,
) -> MultiOutputRegressor:
    """Train a multi-output RandomForest regressor on flattened windows."""
    base = RandomForestRegressor(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
        min_samples_leaf=2,
        max_features="auto",
    )
    model = MultiOutputRegressor(base)
    model.fit(X, y)
    return model


def evaluate(model: MultiOutputRegressor, X: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute common regression metrics on the held-out set."""
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y_true, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, preds))),
        "r2": float(r2_score(y_true, preds)),
    }


def forecast_future(
    model: MultiOutputRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    scaler: Optional[StandardScaler] = None,
) -> pd.DataFrame:
    """Use the latest lookback window to forecast the next horizon days."""
    latest_window = df[feature_cols].iloc[-lookback:].values.flatten().reshape(1, -1)
    if scaler is not None:
        latest_window = scaler.transform(latest_window)
    forecast_values = model.predict(latest_window)[0]
    future_dates = pd.date_range(df.iloc[-1]["date"] + pd.Timedelta(days=1), periods=horizon)
    return pd.DataFrame({"date": future_dates, f"predicted_{target_col}": forecast_values})


def walk_forward_backtest(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    lookback: int,
    horizon: int,
    n_estimators: int,
    random_state: int,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """Perform walk-forward backtesting over the full series.

    For each cutoff point, the model trains on all data up to that date and
    forecasts the next ``horizon`` days. Metrics are computed against the
    realized prices for those future days.
    """

    step = step or horizon
    cutoffs = range(lookback + horizon, len(df) - horizon + 1, step)
    records = []

    for cutoff in cutoffs:
        train_df = df.iloc[:cutoff]
        X_train, y_train = build_sliding_windows(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            horizon=horizon,
        )

        if len(X_train) == 0:
            continue

        scaler = StandardScaler().fit(X_train)
        model = train_model(
            scaler.transform(X_train),
            y_train,
            random_state=random_state,
            n_estimators=n_estimators,
        )

        latest_window = train_df[feature_cols].iloc[-lookback:].values.flatten().reshape(1, -1)
        preds = model.predict(scaler.transform(latest_window))[0]
        actuals = df[target_col].iloc[cutoff : cutoff + horizon].values

        records.append(
            {
                "cutoff_date": train_df.iloc[-1]["date"],
                "mae": mean_absolute_error(actuals, preds),
                "rmse": float(np.sqrt(mean_squared_error(actuals, preds))),
                "r2": r2_score(actuals, preds),
            }
        )

    return pd.DataFrame(records)


def run_pipeline(config: ForecastConfig) -> Tuple[MultiOutputRegressor, dict, pd.DataFrame]:
    raw = load_price_data(config.data_path, config.date_col, config.price_col)
    enriched = add_indicators(raw, config.price_col)

    feature_cols = default_feature_columns(config.price_col)

    X, y = build_sliding_windows(
        enriched,
        feature_cols=feature_cols,
        target_col=config.price_col,
        lookback=config.lookback,
        horizon=config.horizon,
    )

    if len(X) == 0:
        raise ValueError("Not enough data to build any training windows. Increase your dataset.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = train_model(
        X_train,
        y_train,
        random_state=config.random_state,
        n_estimators=config.n_estimators,
    )

    metrics = evaluate(model, X_test, y_test)

    # Fit on full data for the freshest forecast
    full_model = train_model(
        X_scaled,
        y,
        random_state=config.random_state,
        n_estimators=config.n_estimators,
    )
    forecast_df = forecast_future(
        full_model,
        enriched,
        feature_cols=feature_cols,
        target_col=config.price_col,
        lookback=config.lookback,
        horizon=config.horizon,
        scaler=scaler,
    )
    return full_model, metrics, forecast_df


def run_backtest(config: ForecastConfig) -> pd.DataFrame:
    raw = load_price_data(config.data_path, config.date_col, config.price_col)
    enriched = add_indicators(raw, config.price_col)

    backtest_df = walk_forward_backtest(
        enriched,
        feature_cols=default_feature_columns(config.price_col),
        target_col=config.price_col,
        lookback=config.lookback,
        horizon=config.horizon,
        n_estimators=config.n_estimators,
        random_state=config.random_state,
    )

    if backtest_df.empty:
        raise ValueError("Backtest produced no results. Ensure the dataset is long enough.")

    return backtest_df


def parse_args() -> ForecastConfig:
    parser = argparse.ArgumentParser(description="Train a baseline stock forecaster")
    parser.add_argument("--data", type=Path, required=True, help="Path to CSV with price data")
    parser.add_argument("--date-col", type=str, default="date", help="Date column name")
    parser.add_argument("--price-col", type=str, default="close", help="Price column name")
    parser.add_argument("--lookback", type=int, default=60, help="Days of history per sample")
    parser.add_argument("--horizon", type=int, default=20, help="Days to predict")
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees for the RandomForest base model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forecast.csv"),
        help="Where to write the forecast CSV",
    )
    parser.add_argument(
        "--backtest-output",
        type=Path,
        default=None,
        help="Optional path to write walk-forward backtest metrics",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model reproducibility",
    )
    args = parser.parse_args()

    return ForecastConfig(
        data_path=args.data,
        date_col=args.date_col,
        price_col=args.price_col,
        lookback=args.lookback,
        horizon=args.horizon,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        output_path=args.output,
        backtest_output=args.backtest_output,
    )


def main() -> None:
    config = parse_args()
    model, metrics, forecast_df = run_pipeline(config)

    print("Validation metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

    forecast_df.to_csv(config.output_path, index=False)
    print(f"Saved {len(forecast_df)}-day forecast to {config.output_path}")

    if config.backtest_output:
        backtest_df = run_backtest(config)
        backtest_df.to_csv(config.backtest_output, index=False)
        print("Backtest summary (mean over folds):")
        print(backtest_df[["mae", "rmse", "r2"]].mean().to_string())
        print(f"Saved walk-forward backtest metrics to {config.backtest_output}")


if __name__ == "__main__":
    main()
