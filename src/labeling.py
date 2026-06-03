from __future__ import annotations

import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import FINAL_DATASET_PATH, MARKET_FEATURES_PATH
from src.data_loader import read_csv


REGIME_LABELS = ["Bull", "Bear", "Sideways", "High Volatility", "Recovery"]
HMM_FEATURE_COLUMNS = [
    "market_return_mean",
    "market_volatility_mean",
    "market_rsi_mean",
    "market_macd_hist_mean",
    "advance_ratio",
    "avg_drawdown",
    "vix",
    "vix_change",
    "vix_zscore_60",
    "rolling_return_5",
    "rolling_return_20",
]


def _compounded_return(values: pd.Series) -> float:
    return float((1 + values).prod() - 1)


def add_regime_labels(market_features: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "market_return_mean",
        "market_volatility_mean",
        "market_macd_hist_mean",
        "advance_ratio",
        "avg_drawdown",
        "vix",
    }
    missing = sorted(required - set(market_features.columns))
    if missing:
        raise ValueError(f"Market features are missing required columns: {missing}")

    frame = market_features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["rolling_return_5"] = (
        frame["market_return_mean"].rolling(window=5, min_periods=1).apply(_compounded_return)
    )
    frame["rolling_return_20"] = (
        frame["market_return_mean"].rolling(window=20, min_periods=1).apply(_compounded_return)
    )
    frame["previous_20d_return"] = frame["rolling_return_20"].shift(5)

    high_volatility = (frame["vix"] >= 25) | (frame["market_volatility_mean"] >= 0.03)
    bear = (
        (frame["rolling_return_20"] <= -0.05)
        | ((frame["advance_ratio"] <= 0.35) & (frame["rolling_return_5"] < 0))
    )
    recovery = (
        (frame["previous_20d_return"] <= -0.03)
        & (frame["rolling_return_5"] >= 0.015)
        & (frame["market_macd_hist_mean"] > 0)
        & (frame["vix"] < 25)
    )
    bull = (
        (frame["rolling_return_20"] >= 0.04)
        & (frame["advance_ratio"] >= 0.55)
        & (frame["vix"] < 25)
    )

    frame["regime_current"] = "Sideways"
    frame.loc[bull, "regime_current"] = "Bull"
    frame.loc[recovery, "regime_current"] = "Recovery"
    frame.loc[bear, "regime_current"] = "Bear"
    frame.loc[high_volatility, "regime_current"] = "High Volatility"

    frame["regime_t_plus_1"] = frame["regime_current"].shift(-1)
    frame["regime_t_plus_5"] = frame["regime_current"].shift(-5)
    return frame


def _add_return_context(market_features: pd.DataFrame) -> pd.DataFrame:
    frame = market_features.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if "rolling_return_5" not in frame.columns:
        frame["rolling_return_5"] = (
            frame["market_return_mean"].rolling(window=5, min_periods=1).apply(_compounded_return)
        )
    if "rolling_return_20" not in frame.columns:
        frame["rolling_return_20"] = (
            frame["market_return_mean"].rolling(window=20, min_periods=1).apply(_compounded_return)
        )
    if "previous_20d_return" not in frame.columns:
        frame["previous_20d_return"] = frame["rolling_return_20"].shift(5)
    return frame


def _map_hmm_states_to_regimes(frame: pd.DataFrame) -> dict[int, str]:
    state_stats = (
        frame.groupby("hmm_state")
        .agg(
            vix=("vix", "mean"),
            volatility=("market_volatility_mean", "mean"),
            return_20=("rolling_return_20", "mean"),
            return_5=("rolling_return_5", "mean"),
            advance_ratio=("advance_ratio", "mean"),
            drawdown=("avg_drawdown", "mean"),
            macd=("market_macd_hist_mean", "mean"),
        )
        .copy()
    )
    mapping: dict[int, str] = {}
    remaining = set(state_stats.index.tolist())

    def assign(label: str, state: int | None) -> None:
        if state is not None and state in remaining:
            mapping[int(state)] = label
            remaining.remove(state)

    if remaining:
        high_vol_score = state_stats.loc[list(remaining), "vix"] + 300 * state_stats.loc[
            list(remaining), "volatility"
        ]
        assign("High Volatility", int(high_vol_score.idxmax()))

    if remaining:
        bear_score = state_stats.loc[list(remaining), "return_20"] + 0.5 * state_stats.loc[
            list(remaining), "advance_ratio"
        ]
        assign("Bear", int(bear_score.idxmin()))

    if remaining:
        bull_score = state_stats.loc[list(remaining), "return_20"] + 0.25 * state_stats.loc[
            list(remaining), "advance_ratio"
        ]
        assign("Bull", int(bull_score.idxmax()))

    if remaining:
        recovery_score = (
            state_stats.loc[list(remaining), "return_5"]
            + state_stats.loc[list(remaining), "macd"]
            - state_stats.loc[list(remaining), "drawdown"].abs()
        )
        assign("Recovery", int(recovery_score.idxmax()))

    for state in list(remaining):
        mapping[int(state)] = "Sideways"
    return mapping


def add_hmm_regime_labels(
    market_features: pd.DataFrame,
    n_states: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    required = {
        "date",
        "market_return_mean",
        "market_volatility_mean",
        "market_rsi_mean",
        "market_macd_hist_mean",
        "advance_ratio",
        "avg_drawdown",
        "vix",
        "vix_change",
        "vix_zscore_60",
    }
    missing = sorted(required - set(market_features.columns))
    if missing:
        raise ValueError(f"Market features are missing required columns for HMM labeling: {missing}")

    frame = _add_return_context(market_features)
    n_components = min(n_states, len(frame))
    features = frame[HMM_FEATURE_COLUMNS]
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    scaled = StandardScaler().fit_transform(imputed)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=500,
        random_state=random_state,
    )
    model.fit(scaled)
    frame["hmm_state"] = model.predict(scaled)
    state_to_regime = _map_hmm_states_to_regimes(frame)
    frame["hmm_state_regime"] = frame["hmm_state"].map(state_to_regime)
    frame["regime_current"] = frame["hmm_state_regime"]
    frame["regime_t_plus_1"] = frame["regime_current"].shift(-1)
    frame["regime_t_plus_5"] = frame["regime_current"].shift(-5)
    return frame


def build_labeled_dataset(
    market_features_path=MARKET_FEATURES_PATH,
    output_path=FINAL_DATASET_PATH,
    method: str = "hmm",
) -> pd.DataFrame:
    market_features = read_csv(market_features_path)
    if method == "hmm":
        labeled = add_hmm_regime_labels(market_features)
    elif method == "rule":
        labeled = add_regime_labels(market_features)
    else:
        raise ValueError("method must be either 'hmm' or 'rule'.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(output_path, index=False)
    return labeled


if __name__ == "__main__":
    result = build_labeled_dataset()
    print(f"Saved {len(result):,} labeled market rows to {FINAL_DATASET_PATH}")
    print("Labeling method: HMM")
    print(result["regime_current"].value_counts().to_string())
