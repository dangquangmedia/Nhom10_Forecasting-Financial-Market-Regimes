from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

import src.config
from src.data_loader import read_csv
from src.modeling import make_time_splits


def _predict_xgboost_artifact(splits, artifact: dict[str, object]) -> tuple[pd.Series, pd.DataFrame]:
    model = artifact["model"]
    encoder = artifact["label_encoder"]
    imputer = artifact["imputer"]
    X_test = imputer.transform(splits.X_test)
    probabilities = model.predict_proba(X_test)
    predicted_codes = probabilities.argmax(axis=1)
    predicted_labels = encoder.inverse_transform(predicted_codes)
    probability_frame = pd.DataFrame(
        probabilities,
        columns=[f"prob_{label}" for label in encoder.classes_],
        index=splits.X_test.index,
    )
    return pd.Series(predicted_labels, index=splits.X_test.index, name="y_pred"), probability_frame


def _classification_report_frame(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = []
    for label, values in report.items():
        if isinstance(values, dict):
            rows.append(
                {
                    "label": label,
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "f1_score": values.get("f1-score"),
                    "support": values.get("support"),
                }
            )
        else:
            rows.append(
                {
                    "label": label,
                    "precision": None,
                    "recall": None,
                    "f1_score": values,
                    "support": None,
                }
            )
    return pd.DataFrame(rows)


def _feature_importance_frame(splits, artifact: dict[str, object]) -> pd.DataFrame:
    model = artifact["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        raise ValueError("XGBoost artifact does not expose feature_importances_.")
    frame = pd.DataFrame(
        {
            "feature": splits.feature_columns,
            "importance": importances,
        }
    )
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def build_xgboost_evaluation_artifacts(
    dataset: pd.DataFrame,
    artifact: dict[str, object],
) -> dict[str, pd.DataFrame]:
    splits = make_time_splits(dataset)
    y_pred, probability_frame = _predict_xgboost_artifact(splits, artifact)
    y_true = splits.y_test.reset_index(drop=True)
    y_pred_reset = y_pred.reset_index(drop=True)

    predictions = pd.DataFrame(
        {
            "date": splits.test_dates.reset_index(drop=True).dt.strftime("%Y-%m-%d"),
            "y_true": y_true,
            "y_pred": y_pred_reset,
            "confidence": probability_frame.max(axis=1).reset_index(drop=True),
        }
    )
    predictions = pd.concat([predictions, probability_frame.reset_index(drop=True)], axis=1)

    labels = list(artifact["label_encoder"].classes_)
    matrix = confusion_matrix(y_true, y_pred_reset, labels=labels)
    confusion = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion.index.name = "actual"
    confusion.columns.name = "predicted"

    return {
        "predictions": predictions,
        "confusion_matrix": confusion,
        "classification_report": _classification_report_frame(y_true, y_pred_reset),
        "feature_importance": _feature_importance_frame(splits, artifact),
    }


def export_xgboost_evaluation_artifacts(
    dataset_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, pd.DataFrame]:
    from src.backtest import run_regime_backtest
    import shap

    if dataset_path is None:
        dataset_path = src.config.FINAL_DATASET_PATH
    if model_path is None:
        model_path = src.config.XGBOOST_MODEL_PATH

    dataset = read_csv(dataset_path)
    artifact = joblib.load(model_path)
    outputs = build_xgboost_evaluation_artifacts(dataset, artifact)

    # Chạy kiểm thử chiến lược đầu tư vĩ mô
    equity_curves, metrics_df = run_regime_backtest(outputs["predictions"], dataset)
    outputs["backtest_results"] = equity_curves
    outputs["backtest_metrics"] = metrics_df

    # Tính toán trước SHAP values
    model = artifact["model"]
    encoder = artifact["label_encoder"]
    imputer = artifact["imputer"]
    splits = make_time_splits(dataset)
    X_test_imputed = pd.DataFrame(
        imputer.transform(splits.X_test),
        columns=splits.feature_columns
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_imputed)

    # Lưu dữ liệu SHAP
    shap_data = {
        "shap_values": shap_values,
        "X_test_imputed": X_test_imputed,
        "classes": list(encoder.classes_)
    }
    src.config.XGBOOST_SHAP_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(shap_data, src.config.XGBOOST_SHAP_DATA_PATH)
    print(f"Saved SHAP values to {src.config.XGBOOST_SHAP_DATA_PATH}")

    path_map = {
        "predictions": src.config.XGBOOST_TEST_PREDICTIONS_PATH,
        "confusion_matrix": src.config.XGBOOST_CONFUSION_MATRIX_PATH,
        "classification_report": src.config.XGBOOST_CLASSIFICATION_REPORT_PATH,
        "feature_importance": src.config.XGBOOST_FEATURE_IMPORTANCE_PATH,
        "backtest_results": src.config.BACKTEST_RESULTS_PATH,
        "backtest_metrics": src.config.BACKTEST_METRICS_PATH,
    }
    for name, frame in outputs.items():
        path = path_map[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path)
    return outputs


if __name__ == "__main__":
    exported = export_xgboost_evaluation_artifacts()
    for name, frame in exported.items():
        print(f"Saved {name}: {frame.shape}")

