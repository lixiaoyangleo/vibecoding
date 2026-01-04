# vibecoding

Baseline stock time-series forecasting example using a sliding-window RandomForest regressor.

## Requirements
- Python 3.9+
- pandas, numpy, scikit-learn
- pytest (for running the integration test)

## Usage
1. Prepare a CSV with at least two years of daily data containing:
   - `date`: parsable date column.
   - `close`: closing price column (or override with `--price-col`).
2. Run the pipeline:
   ```bash
   python stock_forecast.py \
     --data your_prices.csv \
     --date-col date \
     --price-col close \
     --lookback 60 \
     --horizon 20 \
     --output forecast.csv
   ```
3. (Optional) Add walk-forward backtesting and persist metrics:
   ```bash
   python stock_forecast.py \
     --data your_prices.csv \
     --date-col date \
     --price-col close \
     --lookback 60 \
     --horizon 20 \
     --output forecast.csv \
     --backtest-output backtest.csv
   ```
4. Review validation metrics printed to stdout and the saved `forecast.csv`, which
   contains a 20-day trajectory for the predicted closing price. When enabled,
   `backtest.csv` stores walk-forward fold metrics (MAE/RMSE/R²) with their
   corresponding cutoff dates.

The script trains on sliding windows of historical prices plus simple technical
indicators (returns, moving averages, volatility) and produces a multi-output
forecast for the next month.

## Running tests

The repository includes an integration test that downloads the past two years of
Apple (AAPL) prices from Stooq, fits the model, and produces a 20-day forecast.
Install `pytest`, then run:

```bash
pytest
```
