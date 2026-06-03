import pandas as pd

from src.modeling import (
    FEATURE_EXCLUDE_COLUMNS,
    make_time_splits,
    train_baseline_models,
    train_xgboost_model,
)


def _sample_final_dataset() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    rows = []
    labels = ["Bull", "Bear", "Sideways", "High Volatility", "Recovery"]
    for idx, date in enumerate(dates):
        rows.append(
            {
                "date": date,
                "market_return_mean": 0.001 * ((idx % 5) - 2),
                "market_return_median": 0.001 * ((idx % 5) - 2),
                "market_volatility_mean": 0.01 + idx * 0.0001,
                "market_rsi_mean": 40 + (idx % 30),
                "market_macd_hist_mean": 0.01 * ((idx % 3) - 1),
                "market_volume_change_mean": 0.001 * (idx % 4),
                "advance_ratio": (idx % 10) / 10,
                "decline_ratio": 1 - ((idx % 10) / 10),
                "cross_sectional_volatility": 0.01,
                "avg_drawdown": -0.01 * (idx % 8),
                "vix": 15 + (idx % 20),
                "vix_change": 0.01,
                "vix_ma_5": 15 + (idx % 20),
                "vix_ma_20": 15 + (idx % 20),
                "vix_zscore_60": 0.1,
                "sector_dispersion": 0.02,
                "positive_sector_count": idx % 11,
                "rolling_return_5": 0.001 * idx,
                "rolling_return_20": 0.002 * idx,
                "previous_20d_return": 0.001 * max(idx - 5, 0),
                "hmm_state": idx % 5,
                "hmm_state_regime": labels[idx % len(labels)],
                "regime_current": labels[idx % len(labels)],
                "regime_t_plus_1": labels[(idx + 1) % len(labels)],
                "regime_t_plus_5": labels[(idx + 2) % len(labels)] if idx < 75 else None,
            }
        )
    return pd.DataFrame(rows)


def test_make_time_splits_uses_next_day_target_and_preserves_time_order():
    dataset = _sample_final_dataset()

    splits = make_time_splits(dataset)

    assert splits.X_train.index.max() < splits.X_valid.index.min()
    assert splits.X_valid.index.max() < splits.X_test.index.min()
    assert len(splits.y_train) + len(splits.y_valid) + len(splits.y_test) == 80
    assert "date" not in splits.X_train.columns
    for excluded in FEATURE_EXCLUDE_COLUMNS:
        assert excluded not in splits.X_train.columns


def test_train_baseline_models_returns_metrics_for_logistic_and_random_forest():
    dataset = _sample_final_dataset()
    splits = make_time_splits(dataset, target_column="regime_t_plus_5")

    results, artifacts = train_baseline_models(splits)

    assert set(results["model"]) == {"logistic_regression", "random_forest"}
    assert {"accuracy", "macro_f1", "weighted_f1"}.issubset(results.columns)
    assert "logistic_regression" in artifacts
    assert "random_forest" in artifacts


def test_train_baseline_models_handles_initial_missing_feature_values():
    dataset = _sample_final_dataset()
    dataset.loc[:4, "previous_20d_return"] = None
    splits = make_time_splits(dataset, target_column="regime_t_plus_5")

    results, artifacts = train_baseline_models(splits)

    assert len(results) == 2
    assert "logistic_regression" in artifacts
    assert "random_forest" in artifacts


def test_train_xgboost_model_handles_missing_values_without_sklearn_pipeline_error():
    dataset = _sample_final_dataset()
    dataset.loc[:4, "previous_20d_return"] = None
    splits = make_time_splits(dataset, target_column="regime_t_plus_5")

    results, artifacts = train_xgboost_model(splits)

    assert results["model"].tolist() == ["xgboost"]
    assert "xgboost" in artifacts
    assert {"model", "label_encoder", "imputer"}.issubset(artifacts["xgboost"].keys())
