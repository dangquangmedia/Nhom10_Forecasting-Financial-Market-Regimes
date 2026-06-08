import pandas as pd

from src.evaluation import build_xgboost_evaluation_artifacts
from src.modeling import make_time_splits, train_xgboost_model


def _sample_final_dataset() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=80, freq="B")
    labels = ["Bull", "Bear", "Sideways", "High Volatility", "Recovery"]
    rows = []
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
                "fed_funds_rate": 5.33,
                "previous_20d_return": None if idx < 5 else 0.001 * (idx - 5),
                "regime_current": labels[idx % len(labels)],
                "regime_t_plus_1": labels[(idx + 1) % len(labels)],
                "regime_t_plus_5": labels[(idx + 2) % len(labels)] if idx < 75 else None,
            }
        )
    return pd.DataFrame(rows)


def test_build_xgboost_evaluation_artifacts_returns_dashboard_ready_tables():
    dataset = _sample_final_dataset()
    splits = make_time_splits(dataset)
    _, artifacts = train_xgboost_model(splits)

    outputs = build_xgboost_evaluation_artifacts(dataset, artifacts["xgboost"])

    assert set(outputs) == {
        "predictions",
        "confusion_matrix",
        "classification_report",
        "feature_importance",
    }
    assert {"date", "y_true", "y_pred", "confidence"}.issubset(outputs["predictions"].columns)
    assert outputs["confusion_matrix"].index.name == "actual"
    assert outputs["confusion_matrix"].columns.name == "predicted"
    assert {"label", "precision", "recall", "f1_score", "support"}.issubset(
        outputs["classification_report"].columns
    )
    assert {"feature", "importance"}.issubset(outputs["feature_importance"].columns)


def test_export_xgboost_evaluation_artifacts_writes_files(tmp_path, monkeypatch):
    import joblib
    import src.config
    from src.evaluation import export_xgboost_evaluation_artifacts

    # Mock các đường dẫn lưu báo cáo vào thư mục tạm của test
    monkeypatch.setattr(src.config, "XGBOOST_TEST_PREDICTIONS_PATH", tmp_path / "predictions.csv")
    monkeypatch.setattr(src.config, "XGBOOST_CONFUSION_MATRIX_PATH", tmp_path / "confusion_matrix.csv")
    monkeypatch.setattr(src.config, "XGBOOST_CLASSIFICATION_REPORT_PATH", tmp_path / "classification_report.csv")
    monkeypatch.setattr(src.config, "XGBOOST_FEATURE_IMPORTANCE_PATH", tmp_path / "feature_importance.csv")
    monkeypatch.setattr(src.config, "BACKTEST_RESULTS_PATH", tmp_path / "backtest_results.csv")
    monkeypatch.setattr(src.config, "BACKTEST_METRICS_PATH", tmp_path / "backtest_metrics.csv")
    monkeypatch.setattr(src.config, "XGBOOST_SHAP_DATA_PATH", tmp_path / "xgboost_shap_data.joblib")

    # Giả lập dữ liệu và huấn luyện mô hình
    dataset = _sample_final_dataset()
    splits = make_time_splits(dataset)
    _, artifacts = train_xgboost_model(splits)

    mock_model_path = tmp_path / "xgboost.joblib"
    joblib.dump(artifacts["xgboost"], mock_model_path)

    mock_dataset_path = tmp_path / "final_dataset.csv"
    dataset.to_csv(mock_dataset_path, index=False)

    # Chạy hàm xuất báo cáo kiểm thử
    outputs = export_xgboost_evaluation_artifacts(
        dataset_path=mock_dataset_path,
        model_path=mock_model_path
    )

    # Xác minh các tệp tin được sinh ra đúng cấu trúc
    assert "backtest_results" in outputs
    assert "backtest_metrics" in outputs
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "confusion_matrix.csv").exists()
    assert (tmp_path / "backtest_results.csv").exists()
    assert (tmp_path / "backtest_metrics.csv").exists()
    assert (tmp_path / "xgboost_shap_data.joblib").exists()

