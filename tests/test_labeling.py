import pandas as pd

from src.labeling import REGIME_LABELS, add_hmm_regime_labels, add_regime_labels


def _base_row(date, market_return_mean, vix, advance_ratio, avg_drawdown, macd_hist=0.0):
    return {
        "date": date,
        "market_return_mean": market_return_mean,
        "market_return_median": market_return_mean,
        "market_volatility_mean": 0.01,
        "market_rsi_mean": 55.0,
        "market_macd_hist_mean": macd_hist,
        "market_volume_change_mean": 0.0,
        "advance_ratio": advance_ratio,
        "decline_ratio": 1 - advance_ratio,
        "cross_sectional_volatility": 0.01,
        "avg_drawdown": avg_drawdown,
        "vix": vix,
        "vix_change": 0.0,
        "vix_ma_5": vix,
        "vix_ma_20": vix,
        "vix_zscore_60": 0.0,
        "best_sector_return": market_return_mean,
        "worst_sector_return": market_return_mean,
        "sector_dispersion": 0.0,
        "positive_sector_count": 5,
    }


def test_add_regime_labels_creates_current_and_forward_targets():
    dates = pd.date_range("2020-01-01", periods=30, freq="B")
    rows = [
        _base_row(date, 0.004, 16.0, 0.70, -0.02, macd_hist=0.01)
        for date in dates
    ]

    labeled = add_regime_labels(pd.DataFrame(rows))

    assert set(REGIME_LABELS).issuperset(set(labeled["regime_current"].dropna().unique()))
    assert "regime_t_plus_1" in labeled.columns
    assert "regime_t_plus_5" in labeled.columns
    assert labeled["regime_t_plus_5"].iloc[0] == labeled["regime_current"].iloc[5]
    assert pd.isna(labeled["regime_t_plus_5"].iloc[-1])


def test_add_regime_labels_prioritizes_high_volatility_over_other_rules():
    dates = pd.date_range("2020-01-01", periods=25, freq="B")
    rows = [
        _base_row(date, 0.004, 31.0, 0.80, -0.01, macd_hist=0.02)
        for date in dates
    ]

    labeled = add_regime_labels(pd.DataFrame(rows))

    assert labeled["regime_current"].iloc[-1] == "High Volatility"


def test_add_regime_labels_identifies_bear_and_recovery_patterns():
    dates = pd.date_range("2020-01-01", periods=45, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        if idx < 25:
            rows.append(_base_row(date, -0.004, 20.0, 0.25, -0.12, macd_hist=-0.02))
        else:
            rows.append(_base_row(date, 0.004, 18.0, 0.65, -0.10, macd_hist=0.03))

    labeled = add_regime_labels(pd.DataFrame(rows))

    assert "Bear" in labeled["regime_current"].iloc[20:25].tolist()
    assert "Recovery" in labeled["regime_current"].iloc[30:].tolist()


def test_add_hmm_regime_labels_creates_hidden_states_and_forward_targets():
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    rows = []
    for idx, date in enumerate(dates):
        if idx < 24:
            rows.append(_base_row(date, 0.003, 14.0, 0.70, -0.01, macd_hist=0.02))
        elif idx < 48:
            rows.append(_base_row(date, -0.004, 18.0, 0.25, -0.12, macd_hist=-0.03))
        elif idx < 72:
            rows.append(_base_row(date, 0.0002, 16.0, 0.50, -0.03, macd_hist=0.0))
        elif idx < 96:
            rows.append(_base_row(date, 0.001, 32.0, 0.45, -0.08, macd_hist=-0.01))
        else:
            rows.append(_base_row(date, 0.0025, 17.0, 0.65, -0.09, macd_hist=0.03))

    labeled = add_hmm_regime_labels(pd.DataFrame(rows), random_state=7)

    assert "hmm_state" in labeled.columns
    assert "hmm_state_regime" in labeled.columns
    assert set(labeled["regime_current"].unique()).issubset(set(REGIME_LABELS))
    assert "regime_t_plus_1" in labeled.columns
    assert "regime_t_plus_5" in labeled.columns
    assert labeled["regime_t_plus_1"].iloc[0] == labeled["regime_current"].iloc[1]
    assert labeled["regime_t_plus_5"].iloc[0] == labeled["regime_current"].iloc[5]
