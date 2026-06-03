# RegimeLens AI

RegimeLens AI is a financial market regime prediction dashboard for the S&P 500.
The project uses real OHLCV stock data, company sector metadata, and FRED VIXCLS
to build a daily market-level dataset for machine learning.

Primary model: **XGBoost**.

Comparison models:

- Logistic Regression
- Random Forest
- XGBoost

The dashboard and model outputs are for research and education only. They are not
investment advice.

## Day 1 Scope

Day 1 prepares the project foundation:

- Project folder structure
- Python dependencies
- Raw data naming convention
- Data cleaning pipeline
- Merge pipeline for OHLCV + sectors + VIX
- Unit tests for preprocessing behavior

## Raw Data Files

Download the datasets and place them in `data/raw/` with these names:

```text
data/raw/sp500_ohlcv.csv
data/raw/sp500_companies.csv
data/raw/vixcls.csv
```

Expected minimum columns:

```text
sp500_ohlcv.csv:
date, ticker or symbol, open, high, low, close, volume

sp500_companies.csv:
symbol or ticker, sector or gics_sector

vixcls.csv:
date or observation_date, vixcls or vix
```

The preprocessing code accepts common capitalization variants such as `Date`,
`Symbol`, `Close`, and `VIXCLS`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Day 1 Pipeline

After placing the three raw CSV files in `data/raw/`, run:

```bash
python -m src.data_loader
```

Expected output:

```text
data/processed/sp500_clean.csv
```

This file contains cleaned and merged rows with:

```text
date, ticker, open, high, low, close, volume, sector, vix
```

## Run Tests

```bash
pytest -q
```

## Run Day 2 Feature Engineering

After Day 1 creates `data/processed/sp500_clean.csv`, run:

```bash
python -m src.features
```

Expected outputs:

```text
data/processed/stock_features.csv
data/processed/market_features.csv
```

`stock_features.csv` contains ticker-level technical features:

- returns
- log returns
- rolling volatility
- moving averages
- RSI
- MACD
- drawdown

`market_features.csv` contains one row per trading day for model training:

- market return mean and median
- market volatility mean
- market breadth via advance and decline ratios
- cross-sectional volatility
- sector return columns
- best/worst sector return
- sector dispersion
- positive sector count
- VIX change, moving averages, and z-score

## Next Steps

Day 3 creates HMM-based regime labels:

- Bull
- Bear
- Sideways
- High Volatility
- Recovery

The main training target is `regime_t_plus_1` for next-day regime prediction
with XGBoost as the primary model.

Run:

```bash
python -m src.labeling
```

Expected output:

```text
data/processed/final_dataset.csv
```

The final dataset includes:

```text
regime_current
regime_t_plus_1
regime_t_plus_5
hmm_state
hmm_state_regime
rolling_return_5
rolling_return_20
previous_20d_return
```

`regime_t_plus_1` is the main target for model training. `regime_t_plus_5` is
kept for optional short-horizon comparison, but it is not the default training
target.

## Run Day 4 and Day 5 Model Training

Run:

```bash
python -m src.modeling
```

This trains:

- Logistic Regression baseline
- Random Forest baseline
- XGBoost primary model

Expected model artifacts:

```text
data/models/logistic_regression.joblib
data/models/random_forest.joblib
data/models/xgboost.joblib
```

Expected metrics output:

```text
reports/model_results.csv
```

The split is chronological:

```text
Train + validation: 2021-06-01 to 2025-06-04
Test: 2025-06-05 to 2026-02-19
```

Because the available VIX file starts at 2021-06-01, the final training dataset
also starts at 2021-06-01. HMM hidden states are used to create labels, but
`hmm_state` and `hmm_state_regime` are excluded from model features to avoid
label leakage. The latest test window may be class-imbalanced, so `macro_f1`,
`bear_recall`, and `high_volatility_recall` should be interpreted alongside the
class distribution.

## Export Day 5 XGBoost Evaluation Artifacts

Run:

```bash
python -m src.evaluation
```

Expected outputs:

```text
reports/xgboost_test_predictions.csv
reports/xgboost_confusion_matrix.csv
reports/xgboost_classification_report.csv
reports/xgboost_feature_importance.csv
```

These files are dashboard-ready. The prediction file includes the true test
label, predicted label, model confidence, and per-regime probabilities. The
feature importance file is used to explain why XGBoost made its predictions.
