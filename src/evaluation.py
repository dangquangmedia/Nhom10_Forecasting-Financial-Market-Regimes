from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.config import (
    FINAL_DATASET_PATH,
    XGBOOST_CLASSIFICATION_REPORT_PATH,
    XGBOOST_CONFUSION_MATRIX_PATH,
    XGBOOST_FEATURE_IMPORTANCE_PATH,
    XGBOOST_MODEL_PATH,
    XGBOOST_TEST_PREDICTIONS_PATH,
)
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
    dataset_path: Path = FINAL_DATASET_PATH,
    model_path: Path = XGBOOST_MODEL_PATH,
) -> dict[str, pd.DataFrame]:
    dataset = read_csv(dataset_path)
    artifact = joblib.load(model_path)
    outputs = build_xgboost_evaluation_artifacts(dataset, artifact)
    path_map = {
        "predictions": XGBOOST_TEST_PREDICTIONS_PATH,
        "confusion_matrix": XGBOOST_CONFUSION_MATRIX_PATH,
        "classification_report": XGBOOST_CLASSIFICATION_REPORT_PATH,
        "feature_importance": XGBOOST_FEATURE_IMPORTANCE_PATH,
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

