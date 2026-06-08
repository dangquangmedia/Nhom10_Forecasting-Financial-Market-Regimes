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
    min_window: int = 252,
    refit_stride: int = 5,
) -> pd.DataFrame:
    import warnings
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
    n_rows = len(frame)
    if n_rows == 0:
        return frame

    # Dynamically adjust min_window for small datasets (e.g. in tests)
    actual_min_window = min(min_window, n_rows // 2)
    if actual_min_window < 10:
        actual_min_window = min(10, n_rows)

    features = frame[HMM_FEATURE_COLUMNS]
    n_components = min(n_states, actual_min_window)

    hmm_states = [0] * n_rows
    hmm_state_regimes = ["Sideways"] * n_rows
    regime_currents = ["Sideways"] * n_rows

    # 1. Warm-up window: fit HMM on 0..actual_min_window
    sub_features_init = features.iloc[:actual_min_window]
    imputer_init = SimpleImputer(strategy="median")
    scaler_init = StandardScaler()
    scaled_init = scaler_init.fit_transform(imputer_init.fit_transform(sub_features_init))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model_init = GaussianHMM(
            n_components=n_components,
            covariance_type="full",
            n_iter=150,
            random_state=random_state,
        )
        model_init.fit(scaled_init)

    states_init = model_init.predict(scaled_init)
    sub_frame_init = frame.iloc[:actual_min_window].copy()
    sub_frame_init["hmm_state"] = states_init
    state_to_regime_init = _map_hmm_states_to_regimes(sub_frame_init)

    for i in range(actual_min_window):
        hmm_states[i] = int(states_init[i])
        hmm_state_regimes[i] = state_to_regime_init.get(states_init[i], "Sideways")
        regime_currents[i] = hmm_state_regimes[i]

    active_model = model_init
    active_imputer = imputer_init
    active_scaler = scaler_init
    active_state_to_regime = state_to_regime_init

    # 2. Expanding window with refit stride
    for t in range(actual_min_window, n_rows):
        sub_features = features.iloc[:t+1]
        is_refit_step = ((t - actual_min_window) % refit_stride == 0) or (t == actual_min_window)

        if is_refit_step:
            active_imputer = SimpleImputer(strategy="median")
            active_scaler = StandardScaler()
            imputed_sub = active_imputer.fit_transform(sub_features)
            scaled_sub = active_scaler.fit_transform(imputed_sub)

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                active_model = GaussianHMM(
                    n_components=n_components,
                    covariance_type="full",
                    n_iter=150,
                    random_state=random_state,
                )
                active_model.fit(scaled_sub)

            states_sub = active_model.predict(scaled_sub)
            sub_frame = frame.iloc[:t+1].copy()
            sub_frame["hmm_state"] = states_sub
            active_state_to_regime = _map_hmm_states_to_regimes(sub_frame)
        else:
            imputed_sub = active_imputer.transform(sub_features)
            scaled_sub = active_scaler.transform(imputed_sub)
            states_sub = active_model.predict(scaled_sub)

        last_state = int(states_sub[-1])
        hmm_states[t] = last_state
        hmm_state_regimes[t] = active_state_to_regime.get(last_state, "Sideways")
        regime_currents[t] = hmm_state_regimes[t]

    frame["hmm_state"] = hmm_states
    frame["hmm_state_regime"] = hmm_state_regimes
    frame["regime_current"] = regime_currents
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
