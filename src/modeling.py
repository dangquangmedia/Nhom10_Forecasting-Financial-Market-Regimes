from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import (
    FINAL_DATASET_PATH,
    LOGISTIC_REGRESSION_MODEL_PATH,
    MODEL_RESULTS_PATH,
    RANDOM_FOREST_MODEL_PATH,
    RANDOM_STATE,
    XGBOOST_MODEL_PATH,
)
from src.data_loader import read_csv


FEATURE_EXCLUDE_COLUMNS = {
    "date",
    "hmm_state",
    "hmm_state_regime",
    "regime_current",
    "regime_t_plus_1",
    "regime_t_plus_5",
}


@dataclass(frozen=True)
class TimeSplits:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dates: pd.Series
    valid_dates: pd.Series
    test_dates: pd.Series
    feature_columns: list[str]


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [column for column in frame.columns if column not in FEATURE_EXCLUDE_COLUMNS]
    numeric_columns = frame[candidates].select_dtypes(include=["number", "bool"]).columns.tolist()
    return numeric_columns


def make_time_splits(
    dataset: pd.DataFrame,
    target_column: str = "regime_t_plus_1",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> TimeSplits:
    frame = dataset.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.dropna(subset=[target_column]).sort_values("date").reset_index(drop=True)

    feature_columns = _numeric_feature_columns(frame)
    if not feature_columns:
        raise ValueError("No numeric feature columns available for model training.")

    n_rows = len(frame)
    train_end = int(n_rows * train_ratio)
    valid_end = int(n_rows * (train_ratio + valid_ratio))
    if train_end == 0 or valid_end <= train_end or valid_end >= n_rows:
        raise ValueError("Not enough rows for train/validation/test split.")

    X = frame[feature_columns]
    y = frame[target_column]
    dates = frame["date"]
    return TimeSplits(
        X_train=X.iloc[:train_end],
        y_train=y.iloc[:train_end],
        X_valid=X.iloc[train_end:valid_end],
        y_valid=y.iloc[train_end:valid_end],
        X_test=X.iloc[valid_end:],
        y_test=y.iloc[valid_end:],
        train_dates=dates.iloc[:train_end],
        valid_dates=dates.iloc[train_end:valid_end],
        test_dates=dates.iloc[valid_end:],
        feature_columns=feature_columns,
    )


def _metrics_for_predictions(model_name: str, y_true: pd.Series, y_pred) -> dict[str, float | str]:
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "bear_recall": recall_score(
            y_true == "Bear",
            pd.Series(y_pred, index=y_true.index) == "Bear",
            zero_division=0,
        ),
        "high_volatility_recall": recall_score(
            y_true == "High Volatility",
            pd.Series(y_pred, index=y_true.index) == "High Volatility",
            zero_division=0,
        ),
    }


def train_baseline_models(splits: TimeSplits) -> tuple[pd.DataFrame, dict[str, object]]:
    models = {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                        solver="saga",
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }
    return _fit_and_evaluate_models(models, splits)


def _fit_and_evaluate_models(
    models: dict[str, object],
    splits: TimeSplits,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    artifacts = {}
    X_train_full = pd.concat([splits.X_train, splits.X_valid])
    y_train_full = pd.concat([splits.y_train, splits.y_valid])

    for model_name, model in models.items():
        model.fit(X_train_full, y_train_full)
        y_pred = model.predict(splits.X_test)
        rows.append(_metrics_for_predictions(model_name, splits.y_test, y_pred))
        artifacts[model_name] = model

    return pd.DataFrame(rows), artifacts


def train_xgboost_model(splits: TimeSplits) -> tuple[pd.DataFrame, dict[str, object]]:
    from xgboost import XGBClassifier

    encoder = LabelEncoder()
    X_train_full = pd.concat([splits.X_train, splits.X_valid])
    y_train_full = encoder.fit_transform(pd.concat([splits.y_train, splits.y_valid]))
    y_test_encoded = encoder.transform(splits.y_test)

    imputer = SimpleImputer(strategy="median")
    X_train_full_imputed = imputer.fit_transform(X_train_full)
    X_test_imputed = imputer.transform(splits.X_test)
    model = XGBClassifier(
        objective="multi:softprob",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_full_imputed, y_train_full)
    y_pred_encoded = model.predict(X_test_imputed)
    y_pred = encoder.inverse_transform(y_pred_encoded)
    y_test = pd.Series(encoder.inverse_transform(y_test_encoded), index=splits.y_test.index)
    metrics = pd.DataFrame([_metrics_for_predictions("xgboost", y_test, y_pred)])
    return metrics, {"xgboost": {"model": model, "label_encoder": encoder, "imputer": imputer}}


def save_artifacts(artifacts: dict[str, object]) -> None:
    path_map = {
        "logistic_regression": LOGISTIC_REGRESSION_MODEL_PATH,
        "random_forest": RANDOM_FOREST_MODEL_PATH,
        "xgboost": XGBOOST_MODEL_PATH,
    }
    for model_name, artifact in artifacts.items():
        path = path_map[model_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)


def train_all_models(
    dataset_path: Path = FINAL_DATASET_PATH,
    results_path: Path = MODEL_RESULTS_PATH,
) -> pd.DataFrame:
    dataset = read_csv(dataset_path)
    splits = make_time_splits(dataset)
    baseline_results, baseline_artifacts = train_baseline_models(splits)
    xgboost_results, xgboost_artifacts = train_xgboost_model(splits)

    results = pd.concat([baseline_results, xgboost_results], ignore_index=True)
    results.insert(1, "target", "regime_t_plus_1")
    results.insert(2, "train_start", splits.train_dates.min().date().isoformat())
    results.insert(3, "train_end", splits.valid_dates.max().date().isoformat())
    results.insert(4, "test_start", splits.test_dates.min().date().isoformat())
    results.insert(5, "test_end", splits.test_dates.max().date().isoformat())

    save_artifacts({**baseline_artifacts, **xgboost_artifacts})
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    return results


if __name__ == "__main__":
    output = train_all_models()
    print(output.round(4).to_string(index=False))
    print(f"Saved model results to {MODEL_RESULTS_PATH}")
